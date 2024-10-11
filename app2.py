import configparser
import json
import os

import chromadb
import flask
import numpy as np
from dotenv import load_dotenv
from functools import wraps
from flask import Flask, jsonify, Response, request, make_response
from vanna.utils import deterministic_uuid

from cache import MemoryCache
from my_vanna import MyVanna

load_dotenv()
app = Flask(__name__, static_url_path='')

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

appConfig = config['appConfig']
appPort = appConfig.getint('appPort')
appName = appConfig['appName']

# 判断数据库类型
dbType = config['dbType']
vectorDbName = dbType['vectorDbName']
# olllama 配置
ollama_config = config['ollama']
ollamaHost = ollama_config['ollamaHost']
ollamaModel = ollama_config['ollamaModel']
chromadbPath = ollama_config['chromadbPath']
chromaClientHost = ollama_config['chromaClientHost']
chromaClientPort = ollama_config.getint('chromaClientPort')
chroma_client = chromadb.HttpClient(host=chromaClientHost, port=chromaClientPort)

# json 文件位置
json_file_path = config['json_file']
json_file_path = json_file_path['json_file_path']


# SETUP
cache = MemoryCache()

initial_prompt = '''
请用中文回答，你只能使用提供给你的sql语句,不可以乱写SQL列名与表名，假如需要加入查询条件，就拼接条件查询，如果列中出现了关键字，
必须使用单引号把关键字括起来，把拼好的sql返回，列中的注释写了列值结果是kv分号分割的键值对，使用case返回注释中的枚举中文值，返回的列名用中文别名，
如果查询数据量超过10条，就只返回前十条,这是{} 数据库
'''.format(dbType['dbType'])

# 使用从配置文件读取的参数
vn = MyVanna(config={
    'model': ollamaModel,
    'ollama_host': ollamaHost,
    'path': chromadbPath,
    'client': chroma_client,
    'initial_prompt': initial_prompt,
    'n_results': 10,  # 设置 ChromaDB 相似性参数n_results只返回一个最相似的结果
    'vectorDbName': vectorDbName,

})

# 判断数据库类型
if dbType['dbType'] == 'mysql':
    # 提取数据库连接参数
    db_config = config['mysql']
    host = db_config['host']
    dbname = db_config['dbname']
    user = db_config['user']
    password = db_config['password']
    port = db_config.getint('port')
    vn.connect_to_mysql(host=host, dbname=dbname, user=user, password=password, port=port)
elif dbType['dbType'] == 'oracle':
    # oracle
    db_config = config['oracle']
    host = db_config['host']
    # orcl
    serviceName = db_config['serviceName']
    dbname = db_config['dbname']
    user = db_config['user']
    password = db_config['password']
    port = db_config.getint('port')
    dsn = f"{host}:{port}/{serviceName}"
    dbClintPath = db_config['dbClintPath']

    vn.connect_to_oracle(dsn=dsn, user=user, password=password, dbClintPath=dbClintPath)
elif dbType['dbType'] == 'sqlserver':
    # mssql
    db_config = config['sqlserver']
    host = db_config['host']
    dbname = db_config['dbname']
    user = db_config['user']
    password = db_config['password']
    port = db_config.getint('port')
    vn.connect_to_mssql(
        odbc_conn_str=f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={host};DATABASE={dbname};UID={user};PWD={password};PORT={port}")

else:
    print("数据库类型未配置")
    exit(-1)


def requires_cache(fields):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            id = request.args.get('id')
            if id is None:
                return jsonify({"type": "error", "error": "未提供ID"})

            for field in fields:
                if cache.get(id=id, field=field) is None:
                    return jsonify({"type": "error", "error": f"未找到{field}"})

            field_values = {field: cache.get(id=id, field=field) for field in fields}
            field_values['id'] = id
            return f(*args, **field_values, **kwargs)

        return decorated

    return decorator


# 移除key
# 移除key
def remove_question_from_json(file_path, question_to_remove):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        return
    try:
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查数据是否是一个列表
        if isinstance(data, list):
            # 过滤掉需要删除的对象
            data = [item for item in data if item.get('question') != question_to_remove]

            # 将结果写回 JSON 文件
            with open(file_path, 'w', encoding='utf-8') as f:
                print("回写成功")
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            print("JSON 文件格式不正确，应该是一个数组。")

    except json.JSONDecodeError:
        print(f"文件 {file_path} 中的 JSON 格式不正确")
    except Exception as e:
        print(f"发生错误: {str(e)}")


# 登录校验
# def requires_auth(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         user = auth.get_user(flask.request)
#
#         if not auth.is_logged_in(user):
#             return jsonify({"type": "not_logged_in", "html": auth.login_form()})
#
#         # Pass the user to the function
#         return f(*args, user=user, **kwargs)
#
#     return decorated


@app.route('/api/v0/generate_questions', methods=['GET'])
# @requires_auth
def generate_questions():
    return jsonify({
        "type": "question_list",
        "questions": vn.generate_questions(),
        "header": "您现在可以问一些问题啦"
    })


# @requires_auth
@app.route('/api/v0/generate_sql', methods=['GET'])
def generate_sql():
    question = flask.request.args.get('question')

    if question is None:
        return jsonify({"type": "error", "error": "没有提供问题"})

    id = cache.generate_id(question=question)
    # sql = llm.api(api_key=api_key, prompt=prompt, parameters=parameters)

    sql = vn.generate_sql(question=question, allow_llm_to_see_data=True)

    cache.set(id=id, field='question', value=question)
    cache.set(id=id, field='sql', value=sql)

    return jsonify(
        {
            "type": "sql",
            "id": id,
            "text": sql,
        })


# @requires_auth
@app.route('/api/v0/run_sql', methods=['GET'])
@requires_cache(['sql'])
def run_sql(id: str, sql: str):
    try:
        df = vn.run_sql(sql=sql)
        cache.set(id=id, field='df', value=df)
        return jsonify(
            {
                "type": "df",
                "id": id,
                "df": df.head(10).to_json(orient='records'),
            })

    except Exception as e:
        return jsonify({"type": "error", "error": str(e)})


# @requires_auth
@app.route('/api/v0/download_csv', methods=['GET'])
@requires_cache(['df'])
def download_csv(id: str, df):
    question = cache.get(id, field='question')

    # 确保 question 不为空
    if not question:
        question = id

    # 生成 CSV 内容
    csv = df.to_csv(index=False, encoding='utf-8-sig')

    # 使用 UTF-8 编码的文件名
    filename = f"{question}.csv".encode('utf-8').decode('latin-1')

    return Response(
        csv.encode('utf-8-sig'),  # 将字符串编码为字节
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )


# @requires_auth
@app.route('/api/v0/generate_plotly_figure', methods=['GET'])
@requires_cache(['df', 'question', 'sql'])
def generate_plotly_figure(id: str, df, question, sql):
    try:
        code = vn.generate_plotly_code(question=question, sql=sql,
                                       df_metadata=f"Running df.dtypes gives:\n {df.dtypes}")
        fig = vn.get_plotly_figure(plotly_code=code, df=df, dark_mode=False)
        fig_json = fig.to_json()
        cache.set(id=id, field='fig_json', value=fig_json)
        return jsonify(
            {
                "type": "plotly_figure",
                "id": id,
                "fig": fig_json,
            })
    except Exception as e:
        # Print the stack trace
        import traceback
        traceback.print_exc()

        return jsonify({"type": "error", "error": str(e)})


# @requires_auth
@app.route('/api/v0/get_training_data', methods=['GET'])
def get_training_data():
    df = vn.get_training_data()
    return jsonify(
        {
            "type": "df",
            "id": "training_data",
            "df": df.head(999999).to_json(orient='records'),
        })


# @requires_auth
@app.route('/api/v0/remove_training_data', methods=['POST'])
def remove_training_data():
    id = flask.request.json.get('id')
    if id is None:
        return jsonify({"type": "error", "error": "ID不能为空"})
    #  TODO 移除json中的问题
    question = cache.get(id, field='question')
    file_path = json_file_path + '/sql.json'
    remove_question_from_json(file_path, question)
    if vn.remove_training_data(id=id):
        return jsonify({"success": True})
    else:
        return jsonify({"type": "error", "error": "不能移除训练数据"})


# @requires_auth
@app.route('/api/v0/train', methods=['POST'])
def add_training_data():
    question = flask.request.json.get('question')
    sql = flask.request.json.get('sql')
    ddl = flask.request.json.get('ddl')
    documentation = flask.request.json.get('documentation')

    try:
        # 初始化切割后的变量
        first_part, second_part = None, None

        # 如果 sql 不为空，进行切割并设置 first_part 为 question，second_part 为 sql
        if sql:
            split_strings = sql.split(',', 1)  # 只分割一次
            first_part = split_strings[0].strip() if len(split_strings) > 0 else None
            second_part = split_strings[1].replace('\n', '').strip() if len(split_strings) > 1 else None
            sql_data = {
                "question": first_part,
                "sql": second_part
            }
            append_to_json_file(json_file_path + '/sql.json', sql_data)
            # 调用 train 并传递 first_part 作为 question, second_part 作为 sql
            id = vn.train(question=first_part, sql=second_part, ddl=None, documentation=None)

        # 如果 ddl 不为空，进行切割并设置 first_part 为 question，second_part 为 ddl
        elif ddl:
            split_strings = ddl.split(',')
            first_part = split_strings[0].strip() if len(split_strings) > 0 else None
            second_part = split_strings[1].strip() if len(split_strings) > 1 else None
            # 写入建表ddl.json
            ddl_data = {
                "question": first_part,
                "ddl": second_part
            }
            append_to_json_file(json_file_path + '/ddl.json', ddl_data)
            # 调用 train 并传递 first_part 作为 question, second_part 作为 ddl
            id = vn.train(question=first_part, sql=None, ddl=second_part, documentation=None)

        # 如果 documentation 不为空，进行切割并设置 first_part 为 question，second_part 为 documentation
        elif documentation:
            split_strings = documentation.split(',')
            first_part = split_strings[0].strip() if len(split_strings) > 0 else None
            second_part = split_strings[1].strip() if len(split_strings) > 1 else None
            # 调用 train 并传递 first_part 作为 question, second_part 作为 documentation
            id = vn.train(question=first_part, sql=None, ddl=None, documentation=second_part)

        return jsonify({"id": id})

    except Exception as e:
        print("TRAINING ERROR", e)
        return jsonify({"type": "error", "error": str(e)})


# @requires_auth
@app.route('/api/v0/generate_followup_questions', methods=['GET'])
@requires_cache(['df', 'question', 'sql'])
def generate_followup_questions(id: str, df, question, sql):
    followup_questions = vn.generate_followup_questions(question=question, sql=sql, df=df)

    cache.set(id=id, field='followup_questions', value=followup_questions)

    return jsonify(
        {
            "type": "question_list",
            "id": id,
            "questions": followup_questions,
            "header": "您可以问一些后续问题:"
        })


# @requires_auth
@app.route('/api/v0/load_question', methods=['GET'])
@requires_cache(['question', 'sql', 'df', 'fig_json', 'followup_questions'])
def load_question(id: str, question, sql, df, fig_json, followup_questions):
    try:
        return jsonify(
            {
                "type": "question_cache",
                "id": id,
                "question": question,
                "sql": sql,
                "df": df.head(10).to_json(orient='records'),
                "fig": fig_json,
                "followup_questions": followup_questions,
            })

    except Exception as e:
        return jsonify({"type": "error", "error": str(e)})


# @requires_auth
@app.route('/api/v0/get_question_history', methods=['GET'])
def get_question_history():
    return jsonify({"type": "question_history", "questions": cache.get_all(field_list=['question'])})


# @app.route("/auth/login", methods=["POST"])
# def login():
#     return auth.login_handler(flask.request)
#
#
# # @requires_auth
# @app.route("/auth/callback", methods=["GET"])
# def callback():
#     return auth.callback_handler(flask.request)
#
#
# # @requires_auth
# @app.route("/auth/logout", methods=["GET"])
# def logout():
#     return auth.logout_handler(flask.request)
#
#


# @requires_auth
@app.route('/')
def root():
    return app.send_static_file('index.html')


def get_databaseVector():
    # 获取表的元信息，包括注释
    df_tables = vn.run_sql(
        f"SELECT TABLE_NAME, TABLE_COMMENT  FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='{dbname}'")

    # 获取列的元信息，包括注释
    df_columns = vn.run_sql(
        f"SELECT TABLE_NAME, COLUMN_NAME, COLUMN_COMMENT, COLUMN_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA='{dbname}'")

    ddl_statements = []
    sql_queries = []
    questions_mapping = {}

    for table_name, table_comment in zip(df_tables['TABLE_NAME'], df_tables['TABLE_COMMENT']):
        if not table_comment:
            continue

        # 开始构建 CREATE TABLE 语句
        ddl = f"CREATE TABLE {table_name} ( "

        # 获取当前表的列信息
        columns_info = df_columns[df_columns['TABLE_NAME'] == table_name]

        column_definitions = []
        select_statements = []

        for _, row in columns_info.iterrows():
            column_name = row['COLUMN_NAME']
            column_comment = row['COLUMN_COMMENT']
            column_type = row['COLUMN_TYPE']

            # 组装列定义
            column_definition = f"  `{column_name}` {column_type}"

            if column_comment:
                column_definition += f" COMMENT '{column_comment}'"
                select_statements.append(f"{column_name} AS `{column_comment}`")
            else:
                select_statements.append(column_name)

            column_definitions.append(column_definition)

        # 拼接所有列定义
        ddl += ", ".join(column_definitions)

        # 添加引擎和字符集定义
        ddl += " ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3"

        # 如果有表注释，则添加
        if table_comment:
            ddl += f" COMMENT='{table_comment}'"

        # 添加分号
        ddl += ";"

        # 存储表名、DDL 和问题
        question = f"查询{table_comment}" if table_comment else f"查询{table_name}"
        questions_mapping[table_name] = (ddl, question)

        # 组装 SQL 查询
        sql_query = f"SELECT  " + ",  ".join(select_statements) + f"  FROM {table_name};"

        # 存储查询语句
        questions_mapping[table_name] += (sql_query,)

        # 将 DDL 和 SQL 查询分别存入数组
        ddl_statements.append(ddl)
        sql_queries.append((question, sql_query))

    # 移动到循环外，确保所有表都被处理
    return ddl_statements, sql_queries, questions_mapping


def append_to_json_file(filename, new_data):
    # 检查文件是否存在
    if os.path.exists(filename):
        # 读取现有的文件内容
        with open(filename, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []  # 如果文件为空或格式不正确，初始化为空数组
    else:
        data = []  # 文件不存在时初始化为空数组

    # 追加新的数据到数组
    data.append(new_data)

    # 将更新后的数组写回文件
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# 定义训练数据的函数
def train_data(ddl_statements, sql_queries, questions_mapping):
    print("开始训练数据...")
    try:
        for table_name, (ddl, question, sql_query) in questions_mapping.items():
            try:
                print('问题：' + question + ',sql:' + sql_query);
                # 写入查询sql.json
                sql_data = {
                    "question": question,
                    "sql": sql_query
                }
                append_to_json_file(json_file_path + '/sql.json', sql_data)
                vn.train(question=question, sql=sql_query, ddl=None, documentation=None)
                print('问题：' + question + ',ddl:' + ddl);
                # 写入建表ddl.json
                ddl_data = {
                    "question": question,
                    "ddl": ddl
                }
                append_to_json_file(json_file_path + '/ddl.json', ddl_data)
                vn.train(question=question, sql=sql_query, ddl=ddl, documentation=None)
            except Exception as e:
                print(f"训练 {table_name} 时发生错误: {str(e)}")
    except Exception as e:
        print(f"训练过程中发生错误: {str(e)}")

    print("训练数据完成。");


#
def check_chroma_file(file_path):
    """检查指定路径下的 chroma.sqlite3 文件是否存在，并判断其大小是否大于 170KB。"""
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)  # 获取文件大小（字节）
        return file_size > 170 * 1024  # 返回 True 如果大于 170KB
    return False  # 文件不存在时返回 False


def check_json_file(file_path):
    """检查指定的 JSON 文件是否存在，以及是否有数据。"""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # 返回 True 如果有数据，返回 False 如果没有数据
                return bool(data)
        except json.JSONDecodeError:
            print("JSON 文件格式不正确或为空")
            return False
        except Exception as e:
            print(f"发生错误: {str(e)}")
            return False
    else:
        print("文件不存在")
    # 文件不存在时返回 False
    return False


# json文件中的查询SQL数据
def train_query_json_querySql_file_data(file_path):
    print("开始训练json中的查询SQL数据...")
    try:
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 提取问题和 SQL
        for index, item in enumerate(data):
            question = item.get("question")
            sql = item.get("sql")

            # 拼接索引到问题前面
            indexed_question = f"{index + 1}: {question}"

            print('问题：' + indexed_question + ', sql: ' + sql)
            vn.train(question=indexed_question, sql=sql, ddl=None, documentation=None)
    except Exception as e:
        print(f"querySQL 训练过程中发生错误: {str(e)}")
    print("训练数据完成。")


# json文件中的DDL数据
def train_query_json_DDL_file_data(file_path):
    print("开始训练json中的查询DDL数据...")
    try:
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        # 提取问题和 SQL
        for item in data:
            question = item.get("question")
            ddl = item.get("ddl")
            print('问题：' + question + ',ddl:' + ddl);
            vn.train(question=question, sql=None, ddl=ddl, documentation=None)
    except Exception as e:
        print(f"DDL SQL训练过程中发生错误: {str(e)}")

    print("训练数据完成。");


# 删除某个key
@app.route('/api/v0/removeChromadbKey', methods=['GET'])
def removeChromadbKey():
    question = request.args.get('question')  # 从查询参数获取 question
    if not question:
        return make_response('缺少必需的参数: question', 400)

    try:
        print('删除key: ' + question)
        remove_question_from_json(json_file_path + '/sql.json', question)
        id = deterministic_uuid(question) + "-sql"
        print("要删除的id:" + id)
        vn.remove_training_data(id=id)
        print('删除key成功: ' + id)
        response = make_response('删除key成功')
        return response
    except Exception as e:
        return make_response(f'删除key成功失败: {str(e)}', 500)


# 清空 chromadb
@app.route('/api/v0/clearChromadb', methods=['GET'])
def clearChromadb():
    try:
        chroma_client.delete_collection(name=vectorDbName + '-sql')
        response = make_response('删除向量查询库成功')
        return response
    except Exception as e:
        return make_response(f'删除向量查询库失败: {str(e)}', 500)


# 问题字符串生成向量
@app.route('/api/v0/genQuestionVector', methods=['GET'])
def genQuestionVector():
    try:
        question = request.args.get('question')
        embeddings = vn.generate_embedding(question)
        # 将 ndarray 转换为列表
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        response = make_response(json.dumps(embeddings_list))
        response.headers['Content-Type'] = 'application/json'
        return response
    except Exception as e:
        return make_response(f'字符串生成向量失败: {str(e)}', 500)


if __name__ == '__main__':
    #
    sqlListCollection = chroma_client.get_collection(name=vectorDbName + '-sql');
    if sqlListCollection is None or sqlListCollection.count() == 0:
        print("未实例化过向量数据库")
        file_path = json_file_path + '/sql.json'
        if check_json_file(file_path):
            print("使用json文件中的查询SQL数据进行训练")
            train_query_json_querySql_file_data(file_path)
            # 使用json文件中的建表DDL数据进行训练
            # ddl_file_path = 'data/ddl.json'
            # if check_json_file(ddl_file_path):
            #     print("使用json文件中的建表DDL数据进行训练")
            #     train_query_json_DDL_file_data(ddl_file_path)
        else:
            # 使用数据库中的表进行训练
            print("json文件不存在数据，未进行训练，请检查数据文件")
            # ddl_statements, sql_queries, questions_mapping = get_databaseVector()
            # train_data(ddl_statements, sql_queries, questions_mapping)
    else:
        print("已经实例化过向量数据库，直接使用不在实例化")

    print(f"启动{appName}应用， 端口 {appPort}")
    app.run(debug=True, host='0.0.0.0', port=appPort)
    print("启动成功...")
