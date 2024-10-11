import os
import re
from typing import Union

import pandas as pd
from vanna import ValidationError
from vanna.exceptions import ImproperlyConfigured, DependencyError

from vanna.ollama import Ollama
from vanna.types import TrainingPlan, TrainingPlanItem

from my_chromadb_vector import My_ChromaDB_VectorStore


class MyVanna(My_ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        My_ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

    # 生成后续关联问题
    def generate_followup_questions(
            self, question: str, sql: str, df: pd.DataFrame, n_questions: int = 5, **kwargs
    ) -> list:

        message_log = [
            self.system_message(
                f"你是一个乐于助人的数据助手。回答使用中文，用户提出了问题：“{question}”\n\n此问题的 SQL 查询为：{sql}\n\n以下是包含查询结果的 pandas DataFrame: \n{df.to_markdown()}\n\n"
            ),
            self.user_message(
                f"回答使用中文，生成用户可能就此数据询问的 {n_questions} 个后续问题列表。用问题列表回复，每行一个。不要用任何解释来回答——只回答问题。请记住，应该有一个可以从问题生成的明确 SQL 查询。最好是可以在此对话上下文之外回答的问题。最好是稍微修改生成的 SQL 查询以允许更深入地挖掘数据的问题。每个问题都将变成一个按钮，用户可以单击该按钮来生成新的 SQL 查询，因此不要使用“示例”类型的问题。每个问题都必须与实例化的 SQL 查询一一对应." +
                self._response_language()
            ),
        ]

        llm_response = self.submit_prompt(message_log, **kwargs)

        numbers_removed = re.sub(r"^\d+\.\s*", "", llm_response, flags=re.MULTILINE)
        return numbers_removed.split("\n")

    # 生成摘要
    def generate_summary(self, question: str, df: pd.DataFrame, **kwargs) -> str:
        message_log = [
            self.system_message(
                f"你是一个乐于助人的数据助手。回答使用中文，用户提出了问题：“{question}”\in\以下是包含查询结果的 pandas DataFrame：\in{df.to_markdown()}\n\n"
            ),
            self.user_message(
                "。回答使用中文，根据所问问题简要总结数据。除了总结之外，不要提供任何其他解释。" +
                self._response_language()
            ),
        ]

        summary = self.submit_prompt(message_log, **kwargs)
        return summary

    # 获取SQL提示词
    def get_sql_prompt(
            self,
            initial_prompt: str,
            question: str,
            question_sql_list: list,
            ddl_list: list,
            doc_list: list,
            **kwargs,
    ):

        if initial_prompt is None:
            initial_prompt = f"你是一个 {self.dialect} 专家,用中文回答. " + \
                             "请帮助生成 SQL 查询来回答问题。您的回复应仅基于给定的上下文，并遵循回复指南和格式说明. "

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt += (
            " == =回复指南 \n "
            "1. 如果提供的上下文足够，请生成有效的 SQL 查询，但不对问题进行任何解释。\n "
            "2. 如果提供的上下文几乎足够，但需要了解特定列中的特定字符串，请生成中间 SQL 查询以查找该列中的不同字符串。在查询前面添加注释，说明 middle_sql \n "
            "3. 如果提供的上下文不足，请解释无法生成的原因。\n "
            "4. 请使用最相关的表。\n "
            "5. 如果之前已经问过并回答过该问题，请准确重复之前的答案。\n "
            f"6. 确保输出 SQL 符合 {self.dialect} 且可执行，并且没有语法错误。\n "
        )

        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))
        return message_log

    # 获取 后续问题提示词
    def get_followup_questions_prompt(
            self,
            question: str,
            question_sql_list: list,
            ddl_list: list,
            doc_list: list,
            **kwargs,
    ) -> list:
        initial_prompt = f"用中文回复，用户最初提出的问题: '{question}': \n\n"

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_sql_to_prompt(
            initial_prompt, question_sql_list, max_tokens=self.max_tokens
        )

        message_log = [self.system_message(initial_prompt)]
        message_log.append(
            self.user_message(
                "用中文回答，生成用户可能就此数据提出的后续问题列表。用问题列表进行回复，每行一个。不要回答任何解释——只回答问题."
            )
        )

        return message_log

    # 生成问题
    def generate_question(self, sql: str, **kwargs) -> str:
        response = self.submit_prompt(
            [
                self.system_message(
                    "用中文回答，用户将向您提供 SQL，您将尝试猜测此查询回答的业务问题是什么。只返回问题，不做任何额外解释。不要在问题中引用表名."
                ),
                self.user_message(sql),
            ],
            **kwargs,
        )

        return response

    # 生成绘图代码
    def generate_plotly_code(
            self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs
    ) -> str:
        if question is not None:
            system_msg = f"以下是一个 pandas DataFrame，其中包含回答用户提出的问题的查询结果: '{question}'"
        else:
            system_msg = "以下是 pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\n DataFrame 是使用此查询生成的: {sql}\n\n"

        system_msg += f"以下是有关生成的 pandas DataFrame 的信息 'df': \n{df_metadata}"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                "你能生成 Python plotly 代码来绘制数据框的结果吗？假设数据位于名为 'df'. 如果数据框中只有一个值，请使用指标。仅使用 Python 代码进行回复。不要回答任何解释——只提供代码."
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)

        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))

    # 训练
    def train(
            self,
            question: str = None,
            sql: str = None,
            ddl: str = None,
            documentation: str = None,
            plan: TrainingPlan = None,
    ) -> str:
        if documentation:
            print("添加 文档....")
            return self.add_documentation(documentation)

        if sql:
            if question is None:
                question = self.generate_question(sql)
            return self.add_question_sql(question=question, sql=sql)

        if ddl:
            print("添加 ddl:", ddl)
            return self.add_ddl(question=question, ddl=ddl)

        if plan:
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
                    self.add_question_sql(question=item.item_name, sql=item.item_value)

    def connect_to_oracle(
            self,
            user: str = None,
            password: str = None,
            dsn: str = None,
            dbClintPath: str = None,
            **kwargs
    ):

        """
        Connect to an Oracle db using oracledb package. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_oracle(
        user="username",
        password="password",
        dns="host:port/sid",
        )
        ```
        Args:
            USER (str): Oracle db user name.
            PASSWORD (str): Oracle db user password.
            DSN (str): Oracle db host ip - host:port/sid.
        """

        try:
            import oracledb
        except ImportError:

            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install oracledb"
            )

        if not dsn:
            dsn = os.getenv("DSN")

        if not dsn:
            raise ImproperlyConfigured("Please set your Oracle dsn which should include host:port/sid")

        if not user:
            user = os.getenv("USER")

        if not user:
            raise ImproperlyConfigured("Please set your Oracle db user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your Oracle db password")

        conn = None
        try:
            oracledb.init_oracle_client(lib_dir=dbClintPath)
            conn = oracledb.connect(
                user=user,
                password=password,
                dsn=dsn,
                **kwargs
            )
        except oracledb.Error as e:
            raise ValidationError(e)

        def run_sql_oracle(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    sql = sql.rstrip()
                    if sql.endswith(
                            ';'):  # fix for a known problem with Oracle db where an extra ; will cause an error.
                        sql = sql[:-1]

                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except oracledb.Error as e:
                    conn.rollback()
                    raise ValidationError(e)

                except Exception as e:
                    conn.rollback()
                    raise e

        self.run_sql_is_set = True
        self.run_sql = run_sql_oracle
