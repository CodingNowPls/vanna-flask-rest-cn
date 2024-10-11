var yr = Object.defineProperty;
var br = (r, e, n) => e in r ? yr(r, e, {enumerable: !0, configurable: !0, writable: !0, value: n}) : r[e] = n;
var tn = (r, e, n) => (br(r, typeof e != "symbol" ? e + "" : e, n), n);
(function () {
    const e = document.createElement("link").relList;
    if (e && e.supports && e.supports("modulepreload")) return;
    for (const o of document.querySelectorAll('link[rel="modulepreload"]')) t(o);
    new MutationObserver(o => {
        for (const i of o) if (i.type === "childList") for (const s of i.addedNodes) s.tagName === "LINK" && s.rel === "modulepreload" && t(s)
    }).observe(document, {childList: !0, subtree: !0});

    function n(o) {
        const i = {};
        return o.integrity && (i.integrity = o.integrity), o.referrerPolicy && (i.referrerPolicy = o.referrerPolicy), o.crossOrigin === "use-credentials" ? i.credentials = "include" : o.crossOrigin === "anonymous" ? i.credentials = "omit" : i.credentials = "same-origin", i
    }

    function t(o) {
        if (o.ep) return;
        o.ep = !0;
        const i = n(o);
        fetch(o.href, i)
    }
})();

function se() {
}

function vr(r, e) {
    for (const n in e) r[n] = e[n];
    return r
}

function rr(r) {
    return r()
}

function xn() {
    return Object.create(null)
}

function ot(r) {
    r.forEach(rr)
}

function dt(r) {
    return typeof r == "function"
}

function xe(r, e) {
    return r != r ? e == e : r !== e || r && typeof r == "object" || typeof r == "function"
}

let Vt;

function or(r, e) {
    return r === e ? !0 : (Vt || (Vt = document.createElement("a")), Vt.href = e, r === Vt.href)
}

function _r(r) {
    return Object.keys(r).length === 0
}

function Gt(r, e, n, t) {
    if (r) {
        const o = ir(r, e, n, t);
        return r[0](o)
    }
}

function ir(r, e, n, t) {
    return r[1] && t ? vr(n.ctx.slice(), r[1](t(e))) : n.ctx
}

function Ut(r, e, n, t) {
    if (r[2] && t) {
        const o = r[2](t(n));
        if (e.dirty === void 0) return o;
        if (typeof o == "object") {
            const i = [], s = Math.max(e.dirty.length, o.length);
            for (let l = 0; l < s; l += 1) i[l] = e.dirty[l] | o[l];
            return i
        }
        return e.dirty | o
    }
    return e.dirty
}

function Zt(r, e, n, t, o, i) {
    if (o) {
        const s = ir(e, n, t, i);
        r.p(s, o)
    }
}

function Wt(r) {
    if (r.ctx.length > 32) {
        const e = [], n = r.ctx.length / 32;
        for (let t = 0; t < n; t++) e[t] = -1;
        return e
    }
    return -1
}

function w(r, e) {
    r.appendChild(e)
}

function U(r, e, n) {
    r.insertBefore(e, n || null)
}

function G(r) {
    r.parentNode && r.parentNode.removeChild(r)
}

function Je(r, e) {
    for (let n = 0; n < r.length; n += 1) r[n] && r[n].d(e)
}

function E(r) {
    return document.createElement(r)
}

function Pt(r) {
    return document.createElementNS("http://www.w3.org/2000/svg", r)
}

function we(r) {
    return document.createTextNode(r)
}

function Z() {
    return we(" ")
}

function nt() {
    return we("")
}

function je(r, e, n, t) {
    return r.addEventListener(e, n, t), () => r.removeEventListener(e, n, t)
}

function v(r, e, n) {
    n == null ? r.removeAttribute(e) : r.getAttribute(e) !== n && r.setAttribute(e, n)
}

function wr(r) {
    let e;
    return {
        p(...n) {
            e = n, e.forEach(t => r.push(t))
        }, r() {
            e.forEach(n => r.splice(r.indexOf(n), 1))
        }
    }
}

function kr(r) {
    return Array.from(r.childNodes)
}

function Ue(r, e) {
    e = "" + e, r.data !== e && (r.data = e)
}

function bt(r, e) {
    r.value = e ?? ""
}

let Dt;

function qt(r) {
    Dt = r
}

function xr() {
    if (!Dt) throw new Error("在组件初始化之外调用的函数");
    return Dt
}

function sr(r) {
    xr().$$.on_mount.push(r)
}

const St = [], At = [];
let Ot = [];
const sn = [], $r = Promise.resolve();
let ln = !1;

function Sr() {
    ln || (ln = !0, $r.then(lr))
}

function an(r) {
    Ot.push(r)
}

function nn(r) {
    sn.push(r)
}

const rn = new Set;
let $t = 0;

function lr() {
    if ($t !== 0) return;
    const r = Dt;
    do {
        try {
            for (; $t < St.length;) {
                const e = St[$t];
                $t++, qt(e), Or(e.$$)
            }
        } catch (e) {
            throw St.length = 0, $t = 0, e
        }
        for (qt(null), St.length = 0, $t = 0; At.length;) At.pop()();
        for (let e = 0; e < Ot.length; e += 1) {
            const n = Ot[e];
            rn.has(n) || (rn.add(n), n())
        }
        Ot.length = 0
    } while (St.length);
    for (; sn.length;) sn.pop()();
    ln = !1, rn.clear(), qt(r)
}

function Or(r) {
    if (r.fragment !== null) {
        r.update(), ot(r.before_update);
        const e = r.dirty;
        r.dirty = [-1], r.fragment && r.fragment.p(r.ctx, e), r.after_update.forEach(an)
    }
}

function Lr(r) {
    const e = [], n = [];
    Ot.forEach(t => r.indexOf(t) === -1 ? e.push(t) : n.push(t)), n.forEach(t => t()), Ot = e
}

const zt = new Set;
let yt;

function Ne() {
    yt = {r: 0, c: [], p: yt}
}

function Qe() {
    yt.r || ot(yt.c), yt = yt.p
}

function M(r, e) {
    r && r.i && (zt.delete(r), r.i(e))
}

function I(r, e, n, t) {
    if (r && r.o) {
        if (zt.has(r)) return;
        zt.add(r), yt.c.push(() => {
            zt.delete(r), t && (n && r.d(1), t())
        }), r.o(e)
    } else t && t()
}

function _e(r) {
    return (r == null ? void 0 : r.length) !== void 0 ? r : Array.from(r)
}

function on(r, e, n) {
    const t = r.$$.props[e];
    t !== void 0 && (r.$$.bound[t] = n, n(r.$$.ctx[t]))
}

function J(r) {
    r && r.c()
}

function F(r, e, n) {
    const {fragment: t, after_update: o} = r.$$;
    t && t.m(e, n), an(() => {
        const i = r.$$.on_mount.map(rr).filter(dt);
        r.$$.on_destroy ? r.$$.on_destroy.push(...i) : ot(i), r.$$.on_mount = []
    }), o.forEach(an)
}

function Y(r, e) {
    const n = r.$$;
    n.fragment !== null && (Lr(n.after_update), ot(n.on_destroy), n.fragment && n.fragment.d(e), n.on_destroy = n.fragment = null, n.ctx = [])
}

function Tr(r, e) {
    r.$$.dirty[0] === -1 && (St.push(r), Sr(), r.$$.dirty.fill(0)), r.$$.dirty[e / 31 | 0] |= 1 << e % 31
}

function $e(r, e, n, t, o, i, s, l = [-1]) {
    const c = Dt;
    qt(r);
    const a = r.$$ = {
        fragment: null,
        ctx: [],
        props: i,
        update: se,
        not_equal: o,
        bound: xn(),
        on_mount: [],
        on_destroy: [],
        on_disconnect: [],
        before_update: [],
        after_update: [],
        context: new Map(e.context || (c ? c.$$.context : [])),
        callbacks: xn(),
        dirty: l,
        skip_bound: !1,
        root: e.target || c.$$.root
    };
    s && s(a.root);
    let d = !1;
    if (a.ctx = n ? n(r, e.props || {}, (g, k, ...x) => {
        const D = x.length ? x[0] : k;
        return a.ctx && o(a.ctx[g], a.ctx[g] = D) && (!a.skip_bound && a.bound[g] && a.bound[g](D), d && Tr(r, g)), k
    }) : [], a.update(), d = !0, ot(a.before_update), a.fragment = t ? t(a.ctx) : !1, e.target) {
        if (e.hydrate) {
            const g = kr(e.target);
            a.fragment && a.fragment.l(g), g.forEach(G)
        } else a.fragment && a.fragment.c();
        e.intro && M(r.$$.fragment), F(r, e.target, e.anchor), lr()
    }
    qt(c)
}

class Se {
    constructor() {
        tn(this, "$$");
        tn(this, "$$set")
    }

    $destroy() {
        Y(this, 1), this.$destroy = se
    }

    $on(e, n) {
        if (!dt(n)) return se;
        const t = this.$$.callbacks[e] || (this.$$.callbacks[e] = []);
        return t.push(n), () => {
            const o = t.indexOf(n);
            o !== -1 && t.splice(o, 1)
        }
    }

    $set(e) {
        this.$$set && !_r(e) && (this.$$.skip_bound = !0, this.$$set(e), this.$$.skip_bound = !1)
    }
}

const Er = "4";
typeof window < "u" && (window.__svelte || (window.__svelte = {v: new Set})).v.add(Er);

function $n(r, e, n) {
    const t = r.slice();
    return t[4] = e[n], t[6] = n, t
}

function Sn(r) {
    let e;
    return {
        c() {
            e = E("span"), e.innerHTML = "", v(e, "class", "inline")
        }, m(n, t) {
            U(n, e, t)
        }, d(n) {
            n && G(e)
        }
    }
}

function On(r) {
    let e, n = r[4] + "", t, o, i, s, l = r[6] < r[0] && Sn();
    return {
        c() {
            e = E("span"), t = we(n), i = Z(), l && l.c(), s = nt(), v(e, "class", o = r[6] < r[0] ? "inline" : "hidden")
        }, m(c, a) {
            U(c, e, a), w(e, t), U(c, i, a), l && l.m(c, a), U(c, s, a)
        }, p(c, a) {
            a & 1 && o !== (o = c[6] < c[0] ? "inline" : "hidden") && v(e, "class", o), c[6] < c[0] ? l || (l = Sn(), l.c(), l.m(s.parentNode, s)) : l && (l.d(1), l = null)
        }, d(c) {
            c && (G(e), G(i), G(s)), l && l.d(c)
        }
    }
}

function Cr(r) {
    let e, n = _e(r[1]), t = [];
    for (let o = 0; o < n.length; o += 1) t[o] = On($n(r, n, o));
    return {
        c() {
            for (let o = 0; o < t.length; o += 1) t[o].c();
            e = nt()
        }, m(o, i) {
            for (let s = 0; s < t.length; s += 1) t[s] && t[s].m(o, i);
            U(o, e, i)
        }, p(o, [i]) {
            if (i & 3) {
                n = _e(o[1]);
                let s;
                for (s = 0; s < n.length; s += 1) {
                    const l = $n(o, n, s);
                    t[s] ? t[s].p(l, i) : (t[s] = On(l), t[s].c(), t[s].m(e.parentNode, e))
                }
                for (; s < t.length; s += 1) t[s].d(1);
                t.length = n.length
            }
        }, i: se, o: se, d(o) {
            o && G(e), Je(t, o)
        }
    }
}

function jr(r, e, n) {
    let {text: t} = e, o = t.split(" "), i = 0;
    return setInterval(() => {
        i < o.length && n(0, i++, i)
    }, 100), r.$$set = l => {
        "text" in l && n(2, t = l.text)
    }, [i, o, t]
}

class Pr extends Se {
    constructor(e) {
        super(), $e(this, e, jr, Cr, xe, {text: 2})
    }
}

function Ln(r, e, n) {
    const t = r.slice();
    return t[5] = e[n], t
}

function Tn(r) {
    let e, n, t, o, i, s = r[5].question + "", l, c, a, d;

    function g() {
        return r[4](r[5])
    }

    return {
        c() {
            e = E("li"), n = E("button"), t = Pt("svg"), o = Pt("path"), i = Z(), l = we(s), c = Z(), v(o, "stroke-linecap", "round"), v(o, "stroke-linejoin", "round"), v(o, "d", "M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z"), v(t, "class", "w-3.5 h-3.5"), v(t, "fill", "none"), v(t, "stroke", "currentColor"), v(t, "stroke-width", "1.5"), v(t, "viewBox", "0 0 24 24"), v(t, "xmlns", "http://www.w3.org/2000/svg"), v(t, "aria-hidden", "true"), v(n, "class", "flex items-center text-left gap-x-3 py-2 px-3 text-sm text-slate-700 rounded-md hover:bg-gray-100 dark:hover:bg-gray-900 dark:text-slate-400 dark:hover:text-slate-300")
        }, m(k, x) {
            U(k, e, x), w(e, n), w(n, t), w(t, o), w(n, i), w(n, l), w(e, c), a || (d = je(n, "click", g), a = !0)
        }, p(k, x) {
            r = k, x & 8 && s !== (s = r[5].question + "") && Ue(l, s)
        }, d(k) {
            k && G(e), a = !1, d()
        }
    }
}

function qr(r) {
    let e, n, t, o, i, s, l, c, a, d, g, k, x, D, L, m, h = _e(r[3]), p = [];
    for (let _ = 0; _ < h.length; _ += 1) p[_] = Tn(Ln(r, h, _));
    return {
        c() {
            e = E("div"), n = E("nav"), t = E("div"), t.innerHTML = '<img class="w-28 h-auto" src="https://img.vanna.ai/vanna-flask.svg" alt="Vanna.AI"/> <div class="lg:hidden"><button type="button" class="w-8 h-8 inline-flex justify-center items-center gap-2 rounded-md text-gray-700 align-middle focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-white focus:ring-blue-600 transition-all dark:text-gray-400 dark:focus:ring-offset-gray-800" data-hs-overlay="#application-sidebar" aria-controls="application-sidebar" aria-label="Toggle navigation"><svg class="w-4 h-4" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M2.146 2.854a.5.5 0 1 1 .708-.708L8 7.293l5.146-5.147a.5.5 0 0 1 .708.708L8.707 8l5.147 5.146a.5.5 0 0 1-.708.708L8 8.707l-5.146 5.147a.5.5 0 0 1-.708-.708L7.293 8 2.146 2.854Z"></path></svg> <span class="sr-only">Sidebar</span></button></div>', o = Z(), i = E("div"), s = E("ul"), l = E("li"), c = E("button"), c.innerHTML = `<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" d="M4.26 10.147a60.436 60.436 0 00-.491 6.347A48.627 48.627 0 0112 20.904a48.627 48.627 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.57 50.57 0 00-2.658-.813A59.905 59.905 0 0112 3.493a59.902 59.902 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.697 50.697 0 0112 13.489a50.702 50.702 0 017.74-3.342M6.75 15a.75.75 0 100-1.5.75.75 0 000 1.5zm0 0v-3.675A55.378 55.378 0 0112 8.443m-7.007 11.55A5.981 5.981 0 006.75 15.75v-1.5"></path></svg>
              训练数据`, a = Z(), d = E("li"), g = E("button"), g.innerHTML = `<svg class="w-3.5 h-3.5" xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" clip-rule="evenodd" d="M8 2C8.47339 2 8.85714 2.38376 8.85714 2.85714V7.14286L13.1429 7.14286C13.6162 7.14286 14 7.52661 14 8C14 8.47339 13.6162 8.85714 13.1429 8.85714L8.85714 8.85715V13.1429C8.85714 13.6162 8.47339 14 8 14C7.52661 14 7.14286 13.6162 7.14286 13.1429V8.85715L2.85714 8.85715C2.38376 8.85715 2 8.4734 2 8.00001C2 7.52662 2.38376 7.14287 2.85714 7.14287L7.14286 7.14286V2.85714C7.14286 2.38376 7.52661 2 8 2Z" fill="currentColor"></path></svg>
              新提问`, k = Z();
            for (let _ = 0; _ < p.length; _ += 1) p[_].c();
            x = Z(), D = E("div"), D.innerHTML = `<div class="py-2.5 px-7"><p class="inline-flex items-center gap-x-2 text-xs text-green-600"><span class="block w-1.5 h-1.5 rounded-full bg-green-600"></span>
            已连接</p></div> <div class="p-4 border-t border-gray-200 dark:border-gray-700"><a class="flex justify-between items-center gap-x-3 py-2 px-3 text-sm text-slate-700 rounded-md hover:bg-gray-100 dark:hover:bg-gray-900 dark:text-slate-400 dark:hover:text-slate-300" href="#replace">注销
            <svg class="w-3.5 h-3.5" xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M10 3.5a.5.5 0 0 0-.5-.5h-8a.5.5 0 0 0-.5.5v9a.5.5 0 0 0 .5.5h8a.5.5 0 0 0 .5-.5v-2a.5.5 0 0 1 1 0v2A1.5 1.5 0 0 1 9.5 14h-8A1.5 1.5 0 0 1 0 12.5v-9A1.5 1.5 0 0 1 1.5 2h8A1.5 1.5 0 0 1 11 3.5v2a.5.5 0 0 1-1 0v-2z"></path><path fill-rule="evenodd" d="M4.146 8.354a.5.5 0 0 1 0-.708l3-3a.5.5 0 1 1 .708.708L5.707 7.5H14.5a.5.5 0 0 1 0 1H5.707l2.147 2.146a.5.5 0 0 1-.708.708l-3-3z"></path></svg></a></div>`, v(t, "class", "flex items-center justify-between py-4 pr-4 pl-7"), v(c, "class", "flex items-center gap-x-3 py-2 px-3 text-sm text-slate-700 rounded-md hover:bg-gray-100 dark:hover:bg-gray-900 dark:text-slate-400 dark:hover:text-slate-300 border-t border-b border-gray-200 dark:border-gray-700 w-full"), v(g, "class", "flex items-center gap-x-3 py-2 px-3 text-sm text-slate-700 rounded-md hover:bg-gray-100 dark:hover:bg-gray-900 dark:text-slate-400 dark:hover:text-slate-300"), v(s, "class", "space-y-1.5 p-4"), v(i, "class", "h-full"), v(D, "class", "mt-auto"), v(n, "class", "hs-accordion-group w-full h-full flex flex-col"), v(n, "data-hs-accordion-always-open", ""), v(e, "id", "application-sidebar"), v(e, "class", "hs-overlay hs-overlay-open:translate-x-0 -translate-x-full transition-all duration-300 transform hidden fixed top-0 left-0 bottom-0 z-[60] w-64 bg-white border-r border-gray-200 overflow-y-auto scrollbar-y lg:block lg:translate-x-0 lg:right-auto lg:bottom-0 dark:scrollbar-y dark:bg-slate-900 dark:border-gray-700")
        }, m(_, A) {
            U(_, e, A), w(e, n), w(n, t), w(n, o), w(n, i), w(i, s), w(s, l), w(l, c), w(s, a), w(s, d), w(d, g), w(s, k);
            for (let P = 0; P < p.length; P += 1) p[P] && p[P].m(s, null);
            w(n, x), w(n, D), L || (m = [je(c, "click", function () {
                dt(r[0]) && r[0].apply(this, arguments)
            }), je(g, "click", function () {
                dt(r[1]) && r[1].apply(this, arguments)
            })], L = !0)
        }, p(_, [A]) {
            if (r = _, A & 12) {
                h = _e(r[3]);
                let P;
                for (P = 0; P < h.length; P += 1) {
                    const T = Ln(r, h, P);
                    p[P] ? p[P].p(T, A) : (p[P] = Tn(T), p[P].c(), p[P].m(s, null))
                }
                for (; P < p.length; P += 1) p[P].d(1);
                p.length = h.length
            }
        }, i: se, o: se, d(_) {
            _ && G(e), Je(p, _), L = !1, ot(m)
        }
    }
}

function Ar(r, e, n) {
    let {getTrainingData: t} = e, {newQuestionPage: o} = e, {loadQuestionPage: i} = e, {questionHistory: s} = e;
    const l = c => {
        i(c.id)
    };
    return r.$$set = c => {
        "getTrainingData" in c && n(0, t = c.getTrainingData), "newQuestionPage" in c && n(1, o = c.newQuestionPage), "loadQuestionPage" in c && n(2, i = c.loadQuestionPage), "questionHistory" in c && n(3, s = c.questionHistory)
    }, [t, o, i, s, l]
}

class Dr extends Se {
    constructor(e) {
        super(), $e(this, e, Ar, qr, xe, {
            getTrainingData: 0,
            newQuestionPage: 1,
            loadQuestionPage: 2,
            questionHistory: 3
        })
    }
}

var Hr = {exports: {}};/*! For license information please see preline.js.LICENSE.txt */
(function (r, e) {
    (function (n, t) {
        r.exports = t()
    })(self, function () {
        return (() => {
            var n = {
                661: (s, l, c) => {
                    function a(L) {
                        return a = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function (m) {
                            return typeof m
                        } : function (m) {
                            return m && typeof Symbol == "function" && m.constructor === Symbol && m !== Symbol.prototype ? "symbol" : typeof m
                        }, a(L)
                    }

                    function d(L, m) {
                        for (var h = 0; h < m.length; h++) {
                            var p = m[h];
                            p.enumerable = p.enumerable || !1, p.configurable = !0, "value" in p && (p.writable = !0), Object.defineProperty(L, p.key, p)
                        }
                    }

                    function g(L, m) {
                        return g = Object.setPrototypeOf || function (h, p) {
                            return h.__proto__ = p, h
                        }, g(L, m)
                    }

                    function k(L, m) {
                        if (m && (a(m) === "object" || typeof m == "function")) return m;
                        if (m !== void 0) throw new TypeError("派生构造函数只能返回对象或未定义");
                        return function (h) {
                            if (h === void 0) throw new ReferenceError("这还没有被初始化——super()还没有被调用");
                            return h
                        }(L)
                    }

                    function x(L) {
                        return x = Object.setPrototypeOf ? Object.getPrototypeOf : function (m) {
                            return m.__proto__ || Object.getPrototypeOf(m)
                        }, x(L)
                    }

                    var D = function (L) {
                        (function (T, u) {
                            if (typeof u != "function" && u !== null) throw new TypeError("超级表达式必须为空或者为函数");
                            T.prototype = Object.create(u && u.prototype, {
                                constructor: {
                                    value: T,
                                    writable: !0,
                                    configurable: !0
                                }
                            }), Object.defineProperty(T, "prototype", {writable: !1}), u && g(T, u)
                        })(P, L);
                        var m, h, p, _, A = (p = P, _ = function () {
                            if (typeof Reflect > "u" || !Reflect.construct || Reflect.construct.sham) return !1;
                            if (typeof Proxy == "function") return !0;
                            try {
                                return Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {
                                })), !0
                            } catch {
                                return !1
                            }
                        }(), function () {
                            var T, u = x(p);
                            if (_) {
                                var f = x(this).constructor;
                                T = Reflect.construct(u, arguments, f)
                            } else T = u.apply(this, arguments);
                            return k(this, T)
                        });

                        function P() {
                            return function (T, u) {
                                if (!(T instanceof u)) throw new TypeError("无法将类作为函数调用")
                            }(this, P), A.call(this, ".hs-accordion")
                        }

                        return m = P, (h = [{
                            key: "init", value: function () {
                                var T = this;
                                document.addEventListener("click", function (u) {
                                    var f = u.target, O = f.closest(T.selector), C = f.closest(".hs-accordion-toggle"),
                                        S = f.closest(".hs-accordion-group");
                                    O && S && C && (T._hideAll(O), T.show(O))
                                })
                            }
                        }, {
                            key: "show", value: function (T) {
                                var u = this;
                                if (T.classList.contains("active")) return this.hide(T);
                                T.classList.add("active");
                                var f = T.querySelector(".hs-accordion-content");
                                f.style.display = "block", f.style.height = 0, setTimeout(function () {
                                    f.style.height = "".concat(f.scrollHeight, "px")
                                }), this.afterTransition(f, function () {
                                    T.classList.contains("active") && (f.style.height = "", u._fireEvent("open", T), u._dispatch("open.hs.accordion", T, T))
                                })
                            }
                        }, {
                            key: "hide", value: function (T) {
                                var u = this, f = T.querySelector(".hs-accordion-content");
                                f.style.height = "".concat(f.scrollHeight, "px"), setTimeout(function () {
                                    f.style.height = 0
                                }), this.afterTransition(f, function () {
                                    T.classList.contains("active") || (f.style.display = "", u._fireEvent("hide", T), u._dispatch("hide.hs.accordion", T, T))
                                }), T.classList.remove("active")
                            }
                        }, {
                            key: "_hideAll", value: function (T) {
                                var u = this, f = T.closest(".hs-accordion-group");
                                f.hasAttribute("data-hs-accordion-always-open") || f.querySelectorAll(this.selector).forEach(function (O) {
                                    T !== O && u.hide(O)
                                })
                            }
                        }]) && d(m.prototype, h), Object.defineProperty(m, "prototype", {writable: !1}), P
                    }(c(765).Z);
                    window.HSAccordion = new D, document.addEventListener("load", window.HSAccordion.init())
                }, 795: (s, l, c) => {
                    function a(m) {
                        return a = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function (h) {
                            return typeof h
                        } : function (h) {
                            return h && typeof Symbol == "function" && h.constructor === Symbol && h !== Symbol.prototype ? "symbol" : typeof h
                        }, a(m)
                    }

                    function d(m, h) {
                        (h == null || h > m.length) && (h = m.length);
                        for (var p = 0, _ = new Array(h); p < h; p++) _[p] = m[p];
                        return _
                    }

                    function g(m, h) {
                        for (var p = 0; p < h.length; p++) {
                            var _ = h[p];
                            _.enumerable = _.enumerable || !1, _.configurable = !0, "value" in _ && (_.writable = !0), Object.defineProperty(m, _.key, _)
                        }
                    }

                    function k(m, h) {
                        return k = Object.setPrototypeOf || function (p, _) {
                            return p.__proto__ = _, p
                        }, k(m, h)
                    }

                    function x(m, h) {
                        if (h && (a(h) === "object" || typeof h == "function")) return h;
                        if (h !== void 0) throw new TypeError("派生构造函数只能返回对象或未定义");
                        return function (p) {
                            if (p === void 0) throw new ReferenceError("这还没有被初始化——super()还没有被调用");
                            return p
                        }(m)
                    }

                    function D(m) {
                        return D = Object.setPrototypeOf ? Object.getPrototypeOf : function (h) {
                            return h.__proto__ || Object.getPrototypeOf(h)
                        }, D(m)
                    }

                    var L = function (m) {
                        (function (u, f) {
                            if (typeof f != "function" && f !== null) throw new TypeError("超级表达式必须为空或者为函数");
                            u.prototype = Object.create(f && f.prototype, {
                                constructor: {
                                    value: u,
                                    writable: !0,
                                    configurable: !0
                                }
                            }), Object.defineProperty(u, "prototype", {writable: !1}), f && k(u, f)
                        })(T, m);
                        var h, p, _, A, P = (_ = T, A = function () {
                            if (typeof Reflect > "u" || !Reflect.construct || Reflect.construct.sham) return !1;
                            if (typeof Proxy == "function") return !0;
                            try {
                                return Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {
                                })), !0
                            } catch {
                                return !1
                            }
                        }(), function () {
                            var u, f = D(_);
                            if (A) {
                                var O = D(this).constructor;
                                u = Reflect.construct(f, arguments, O)
                            } else u = f.apply(this, arguments);
                            return x(this, u)
                        });

                        function T() {
                            return function (u, f) {
                                if (!(u instanceof f)) throw new TypeError("无法将类作为函数调用")
                            }(this, T), P.call(this, "[data-hs-collapse]")
                        }

                        return h = T, (p = [{
                            key: "init", value: function () {
                                var u = this;
                                document.addEventListener("click", function (f) {
                                    var O = f.target.closest(u.selector);
                                    if (O) {
                                        var C = document.querySelectorAll(O.getAttribute("data-hs-collapse"));
                                        u.toggle(C)
                                    }
                                })
                            }
                        }, {
                            key: "toggle", value: function (u) {
                                var f, O = this;
                                u.length && (f = u, function (C) {
                                    if (Array.isArray(C)) return d(C)
                                }(f) || function (C) {
                                    if (typeof Symbol < "u" && C[Symbol.iterator] != null || C["@@iterator"] != null) return Array.from(C)
                                }(f) || function (C, S) {
                                    if (C) {
                                        if (typeof C == "string") return d(C, S);
                                        var j = Object.prototype.toString.call(C).slice(8, -1);
                                        return j === "Object" && C.constructor && (j = C.constructor.name), j === "Map" || j === "Set" ? Array.from(C) : j === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(j) ? d(C, S) : void 0
                                    }
                                }(f) || function () {
                                    throw new TypeError(`传播不可迭代实例的尝试无效.
为了可迭代，非数组对象必须具有 [Symbol.iterator]() 方法.`)
                                }()).forEach(function (C) {
                                    C.classList.contains("hidden") ? O.show(C) : O.hide(C)
                                })
                            }
                        }, {
                            key: "show", value: function (u) {
                                var f = this;
                                u.classList.add("open"), u.classList.remove("hidden"), u.style.height = 0, document.querySelectorAll(this.selector).forEach(function (O) {
                                    u.closest(O.getAttribute("data-hs-collapse")) && O.classList.add("open")
                                }), u.style.height = "".concat(u.scrollHeight, "px"), this.afterTransition(u, function () {
                                    u.classList.contains("open") && (u.style.height = "", f._fireEvent("open", u), f._dispatch("open.hs.collapse", u, u))
                                })
                            }
                        }, {
                            key: "hide", value: function (u) {
                                var f = this;
                                u.style.height = "".concat(u.scrollHeight, "px"), setTimeout(function () {
                                    u.style.height = 0
                                }), u.classList.remove("open"), this.afterTransition(u, function () {
                                    u.classList.contains("open") || (u.classList.add("hidden"), u.style.height = null, f._fireEvent("hide", u), f._dispatch("hide.hs.collapse", u, u), u.querySelectorAll(".hs-mega-menu-content.block").forEach(function (O) {
                                        O.classList.remove("block"), O.classList.add("hidden")
                                    }))
                                }), document.querySelectorAll(this.selector).forEach(function (O) {
                                    u.closest(O.getAttribute("data-hs-collapse")) && O.classList.remove("open")
                                })
                            }
                        }]) && g(h.prototype, p), Object.defineProperty(h, "prototype", {writable: !1}), T
                    }(c(765).Z);
                    window.HSCollapse = new L, document.addEventListener("load", window.HSCollapse.init())
                }, 682: (s, l, c) => {
                    var a = c(714), d = c(765);
                    const g = {
                        historyIndex: -1, addHistory: function (A) {
                            this.historyIndex = A
                        }, existsInHistory: function (A) {
                            return A > this.historyIndex
                        }, clearHistory: function () {
                            this.historyIndex = -1
                        }
                    };

                    function k(A) {
                        return k = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function (P) {
                            return typeof P
                        } : function (P) {
                            return P && typeof Symbol == "function" && P.constructor === Symbol && P !== Symbol.prototype ? "symbol" : typeof P
                        }, k(A)
                    }

                    function x(A) {
                        return function (P) {
                            if (Array.isArray(P)) return D(P)
                        }(A) || function (P) {
                            if (typeof Symbol < "u" && P[Symbol.iterator] != null || P["@@iterator"] != null) return Array.from(P)
                        }(A) || function (P, T) {
                            if (P) {
                                if (typeof P == "string") return D(P, T);
                                var u = Object.prototype.toString.call(P).slice(8, -1);
                                return u === "Object" && P.constructor && (u = P.constructor.name), u === "Map" || u === "Set" ? Array.from(P) : u === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(u) ? D(P, T) : void 0
                            }
                        }(A) || function () {
                            throw new TypeError(`传播不可迭代实例的尝试无效.
为了可迭代，非数组对象必须具有 [Symbol.iterator]() 方法.`)
                        }()
                    }

                    function D(A, P) {
                        (P == null || P > A.length) && (P = A.length);
                        for (var T = 0, u = new Array(P); T < P; T++) u[T] = A[T];
                        return u
                    }

                    function L(A, P) {
                        for (var T = 0; T < P.length; T++) {
                            var u = P[T];
                            u.enumerable = u.enumerable || !1, u.configurable = !0, "value" in u && (u.writable = !0), Object.defineProperty(A, u.key, u)
                        }
                    }

                    function m(A, P) {
                        return m = Object.setPrototypeOf || function (T, u) {
                            return T.__proto__ = u, T
                        }, m(A, P)
                    }

                    function h(A, P) {
                        if (P && (k(P) === "object" || typeof P == "function")) return P;
                        if (P !== void 0) throw new TypeError("派生构造函数只能返回对象或未定义");
                        return function (T) {
                            if (T === void 0) throw new ReferenceError("这还没有被初始化——super()还没有被调用");
                            return T
                        }(A)
                    }

                    function p(A) {
                        return p = Object.setPrototypeOf ? Object.getPrototypeOf : function (P) {
                            return P.__proto__ || Object.getPrototypeOf(P)
                        }, p(A)
                    }

                    var _ = function (A) {
                        (function (S, j) {
                            if (typeof j != "function" && j !== null) throw new TypeError("超级表达式必须为空或者为函数");
                            S.prototype = Object.create(j && j.prototype, {
                                constructor: {
                                    value: S,
                                    writable: !0,
                                    configurable: !0
                                }
                            }), Object.defineProperty(S, "prototype", {writable: !1}), j && m(S, j)
                        })(C, A);
                        var P, T, u, f, O = (u = C, f = function () {
                            if (typeof Reflect > "u" || !Reflect.construct || Reflect.construct.sham) return !1;
                            if (typeof Proxy == "function") return !0;
                            try {
                                return Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {
                                })), !0
                            } catch {
                                return !1
                            }
                        }(), function () {
                            var S, j = p(u);
                            if (f) {
                                var R = p(this).constructor;
                                S = Reflect.construct(j, arguments, R)
                            } else S = j.apply(this, arguments);
                            return h(this, S)
                        });

                        function C() {
                            var S;
                            return function (j, R) {
                                if (!(j instanceof R)) throw new TypeError("无法将类作为函数调用")
                            }(this, C), (S = O.call(this, ".hs-dropdown")).positions = {
                                top: "top",
                                "top-left": "top-start",
                                "top-right": "top-end",
                                bottom: "bottom",
                                "bottom-left": "bottom-start",
                                "bottom-right": "bottom-end",
                                right: "right",
                                "right-top": "right-start",
                                "right-bottom": "right-end",
                                left: "left",
                                "left-top": "left-start",
                                "left-bottom": "left-end"
                            }, S.absoluteStrategyModifiers = function (j) {
                                return [{
                                    name: "applyStyles", fn: function (R) {
                                        var q = (window.getComputedStyle(j).getPropertyValue("--strategy") || "absolute").replace(" ", ""),
                                            B = (window.getComputedStyle(j).getPropertyValue("--adaptive") || "adaptive").replace(" ", "");
                                        R.state.elements.popper.style.position = q, R.state.elements.popper.style.transform = B === "adaptive" ? R.state.styles.popper.transform : null, R.state.elements.popper.style.top = null, R.state.elements.popper.style.bottom = null, R.state.elements.popper.style.left = null, R.state.elements.popper.style.right = null, R.state.elements.popper.style.margin = 0
                                    }
                                }, {name: "computeStyles", options: {adaptive: !1}}]
                            }, S._history = g, S
                        }

                        return P = C, T = [{
                            key: "init", value: function () {
                                var S = this;
                                document.addEventListener("click", function (j) {
                                    var R = j.target, q = R.closest(S.selector), B = R.closest(".hs-dropdown-menu");
                                    if (q && q.classList.contains("open") || S._closeOthers(q), B) {
                                        var N = (window.getComputedStyle(q).getPropertyValue("--auto-close") || "").replace(" ", "");
                                        if ((N == "false" || N == "inside") && !q.parentElement.closest(S.selector)) return
                                    }
                                    q && (q.classList.contains("open") ? S.close(q) : S.open(q))
                                }), document.addEventListener("mousemove", function (j) {
                                    var R = j.target, q = R.closest(S.selector);
                                    if (R.closest(".hs-dropdown-menu"), q) {
                                        var B = (window.getComputedStyle(q).getPropertyValue("--trigger") || "click").replace(" ", "");
                                        if (B !== "hover") return;
                                        q && q.classList.contains("open") || S._closeOthers(q), B !== "hover" || q.classList.contains("open") || /iPad|iPhone|iPod/.test(navigator.platform) || navigator.maxTouchPoints && navigator.maxTouchPoints > 2 && /MacIntel/.test(navigator.platform) || navigator.maxTouchPoints && navigator.maxTouchPoints > 2 && /MacIntel/.test(navigator.platform) || S._hover(R)
                                    }
                                }), document.addEventListener("keydown", this._keyboardSupport.bind(this)), window.addEventListener("resize", function () {
                                    document.querySelectorAll(".hs-dropdown.open").forEach(function (j) {
                                        S.close(j, !0)
                                    })
                                })
                            }
                        }, {
                            key: "_closeOthers", value: function () {
                                var S = this, j = arguments.length > 0 && arguments[0] !== void 0 ? arguments[0] : null,
                                    R = document.querySelectorAll("".concat(this.selector, ".open"));
                                R.forEach(function (q) {
                                    if (!j || j.closest(".hs-dropdown.open") !== q) {
                                        var B = (window.getComputedStyle(q).getPropertyValue("--auto-close") || "").replace(" ", "");
                                        B != "false" && B != "outside" && S.close(q)
                                    }
                                })
                            }
                        }, {
                            key: "_hover", value: function (S) {
                                var j = this, R = S.closest(this.selector);
                                this.open(R), document.addEventListener("mousemove", function q(B) {
                                    B.target.closest(j.selector) && B.target.closest(j.selector) !== R.parentElement.closest(j.selector) || (j.close(R), document.removeEventListener("mousemove", q, !0))
                                }, !0)
                            }
                        }, {
                            key: "close", value: function (S) {
                                var j = this, R = arguments.length > 1 && arguments[1] !== void 0 && arguments[1],
                                    q = S.querySelector(".hs-dropdown-menu"), B = function () {
                                        S.classList.contains("open") || (q.classList.remove("block"), q.classList.add("hidden"), q.style.inset = null, q.style.position = null, S._popper && S._popper.destroy())
                                    };
                                R || this.afterTransition(S.querySelector("[data-hs-dropdown-transition]") || q, function () {
                                    B()
                                }), q.style.margin = null, S.classList.remove("open"), R && B(), this._fireEvent("close", S), this._dispatch("close.hs.dropdown", S, S);
                                var N = q.querySelectorAll(".hs-dropdown.open");
                                N.forEach(function (ie) {
                                    j.close(ie, !0)
                                })
                            }
                        }, {
                            key: "open", value: function (S) {
                                var j = S.querySelector(".hs-dropdown-menu"),
                                    R = (window.getComputedStyle(S).getPropertyValue("--placement") || "").replace(" ", ""),
                                    q = (window.getComputedStyle(S).getPropertyValue("--strategy") || "fixed").replace(" ", ""),
                                    B = ((window.getComputedStyle(S).getPropertyValue("--adaptive") || "adaptive").replace(" ", ""), parseInt((window.getComputedStyle(S).getPropertyValue("--offset") || "10").replace(" ", "")));
                                if (q !== "static") {
                                    S._popper && S._popper.destroy();
                                    var N = (0, a.fi)(S, j, {
                                        placement: this.positions[R] || "bottom-start",
                                        strategy: q,
                                        modifiers: [].concat(x(q !== "fixed" ? this.absoluteStrategyModifiers(S) : []), [{
                                            name: "offset",
                                            options: {offset: [0, B]}
                                        }])
                                    });
                                    S._popper = N
                                }
                                j.style.margin = null, j.classList.add("block"), j.classList.remove("hidden"), setTimeout(function () {
                                    S.classList.add("open")
                                }), this._fireEvent("open", S), this._dispatch("open.hs.dropdown", S, S)
                            }
                        }, {
                            key: "_keyboardSupport", value: function (S) {
                                var j = document.querySelector(".hs-dropdown.open");
                                if (j) return S.keyCode === 27 ? (S.preventDefault(), this._esc(j)) : S.keyCode === 40 ? (S.preventDefault(), this._down(j)) : S.keyCode === 38 ? (S.preventDefault(), this._up(j)) : S.keyCode === 36 ? (S.preventDefault(), this._start(j)) : S.keyCode === 35 ? (S.preventDefault(), this._end(j)) : void this._byChar(j, S.key)
                            }
                        }, {
                            key: "_esc", value: function (S) {
                                this.close(S)
                            }
                        }, {
                            key: "_up", value: function (S) {
                                var j = S.querySelector(".hs-dropdown-menu"),
                                    R = x(j.querySelectorAll("a")).reverse().filter(function (N) {
                                        return !N.disabled
                                    }), q = j.querySelector("a:focus"), B = R.findIndex(function (N) {
                                        return N === q
                                    });
                                B + 1 < R.length && B++, R[B].focus()
                            }
                        }, {
                            key: "_down", value: function (S) {
                                var j = S.querySelector(".hs-dropdown-menu"),
                                    R = x(j.querySelectorAll("a")).filter(function (N) {
                                        return !N.disabled
                                    }), q = j.querySelector("a:focus"), B = R.findIndex(function (N) {
                                        return N === q
                                    });
                                B + 1 < R.length && B++, R[B].focus()
                            }
                        }, {
                            key: "_start", value: function (S) {
                                var j = x(S.querySelector(".hs-dropdown-menu").querySelectorAll("a")).filter(function (R) {
                                    return !R.disabled
                                });
                                j.length && j[0].focus()
                            }
                        }, {
                            key: "_end", value: function (S) {
                                var j = x(S.querySelector(".hs-dropdown-menu").querySelectorAll("a")).reverse().filter(function (R) {
                                    return !R.disabled
                                });
                                j.length && j[0].focus()
                            }
                        }, {
                            key: "_byChar", value: function (S, j) {
                                var R = this, q = x(S.querySelector(".hs-dropdown-menu").querySelectorAll("a")),
                                    B = function () {
                                        return q.findIndex(function (ie, re) {
                                            return ie.innerText.toLowerCase().charAt(0) === j.toLowerCase() && R._history.existsInHistory(re)
                                        })
                                    }, N = B();
                                N === -1 && (this._history.clearHistory(), N = B()), N !== -1 && (q[N].focus(), this._history.addHistory(N))
                            }
                        }, {
                            key: "toggle", value: function (S) {
                                S.classList.contains("open") ? this.close(S) : this.open(S)
                            }
                        }], T && L(P.prototype, T), Object.defineProperty(P, "prototype", {writable: !1}), C
                    }(d.Z);
                    window.HSDropdown = new _, document.addEventListener("load", window.HSDropdown.init())
                }, 284: (s, l, c) => {
                    function a(m) {
                        return a = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function (h) {
                            return typeof h
                        } : function (h) {
                            return h && typeof Symbol == "function" && h.constructor === Symbol && h !== Symbol.prototype ? "symbol" : typeof h
                        }, a(m)
                    }

                    function d(m, h) {
                        (h == null || h > m.length) && (h = m.length);
                        for (var p = 0, _ = new Array(h); p < h; p++) _[p] = m[p];
                        return _
                    }

                    function g(m, h) {
                        for (var p = 0; p < h.length; p++) {
                            var _ = h[p];
                            _.enumerable = _.enumerable || !1, _.configurable = !0, "value" in _ && (_.writable = !0), Object.defineProperty(m, _.key, _)
                        }
                    }

                    function k(m, h) {
                        return k = Object.setPrototypeOf || function (p, _) {
                            return p.__proto__ = _, p
                        }, k(m, h)
                    }

                    function x(m, h) {
                        if (h && (a(h) === "object" || typeof h == "function")) return h;
                        if (h !== void 0) throw new TypeError("派生构造函数只能返回对象或未定义");
                        return function (p) {
                            if (p === void 0) throw new ReferenceError("这还没有被初始化——super()还没有被调用");
                            return p
                        }(m)
                    }

                    function D(m) {
                        return D = Object.setPrototypeOf ? Object.getPrototypeOf : function (h) {
                            return h.__proto__ || Object.getPrototypeOf(h)
                        }, D(m)
                    }

                    var L = function (m) {
                        (function (u, f) {
                            if (typeof f != "function" && f !== null) throw new TypeError("超级表达式必须为空或者为函数");
                            u.prototype = Object.create(f && f.prototype, {
                                constructor: {
                                    value: u,
                                    writable: !0,
                                    configurable: !0
                                }
                            }), Object.defineProperty(u, "prototype", {writable: !1}), f && k(u, f)
                        })(T, m);
                        var h, p, _, A, P = (_ = T, A = function () {
                            if (typeof Reflect > "u" || !Reflect.construct || Reflect.construct.sham) return !1;
                            if (typeof Proxy == "function") return !0;
                            try {
                                return Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {
                                })), !0
                            } catch {
                                return !1
                            }
                        }(), function () {
                            var u, f = D(_);
                            if (A) {
                                var O = D(this).constructor;
                                u = Reflect.construct(f, arguments, O)
                            } else u = f.apply(this, arguments);
                            return x(this, u)
                        });

                        function T() {
                            var u;
                            return function (f, O) {
                                if (!(f instanceof O)) throw new TypeError("无法将类作为函数调用")
                            }(this, T), (u = P.call(this, "[data-hs-overlay]")).openNextOverlay = !1, u
                        }

                        return h = T, (p = [{
                            key: "init", value: function () {
                                var u = this;
                                document.addEventListener("click", function (f) {
                                    var O = f.target.closest(u.selector),
                                        C = f.target.closest("[data-hs-overlay-close]"),
                                        S = f.target.getAttribute("aria-overlay") === "true";
                                    return C ? u.close(C.closest(".hs-overlay.open")) : O ? u.toggle(document.querySelector(O.getAttribute("data-hs-overlay"))) : void (S && u._onBackdropClick(f.target))
                                }), document.addEventListener("keydown", function (f) {
                                    if (f.keyCode === 27) {
                                        var O = document.querySelector(".hs-overlay.open");
                                        if (!O) return;
                                        setTimeout(function () {
                                            O.getAttribute("data-hs-overlay-keyboard") !== "false" && u.close(O)
                                        })
                                    }
                                })
                            }
                        }, {
                            key: "toggle", value: function (u) {
                                u && (u.classList.contains("hidden") ? this.open(u) : this.close(u))
                            }
                        }, {
                            key: "open", value: function (u) {
                                var f = this;
                                if (u) {
                                    var O = document.querySelector(".hs-overlay.open"),
                                        C = this.getClassProperty(u, "--body-scroll", "false") !== "true";
                                    if (O) return this.openNextOverlay = !0, this.close(O).then(function () {
                                        f.open(u), f.openNextOverlay = !1
                                    });
                                    C && (document.body.style.overflow = "hidden"), this._buildBackdrop(u), this._checkTimer(u), this._autoHide(u), u.classList.remove("hidden"), u.setAttribute("aria-overlay", "true"), u.setAttribute("tabindex", "-1"), setTimeout(function () {
                                        u.classList.contains("hidden") || (u.classList.add("open"), f._fireEvent("open", u), f._dispatch("open.hs.overlay", u, u), f._focusInput(u))
                                    }, 50)
                                }
                            }
                        }, {
                            key: "close", value: function (u) {
                                var f = this;
                                return new Promise(function (O) {
                                    u && (u.classList.remove("open"), u.removeAttribute("aria-overlay"), u.removeAttribute("tabindex", "-1"), f.afterTransition(u, function () {
                                        u.classList.contains("open") || (u.classList.add("hidden"), f._destroyBackdrop(), f._fireEvent("close", u), f._dispatch("close.hs.overlay", u, u), document.body.style.overflow = "", O(u))
                                    }))
                                })
                            }
                        }, {
                            key: "_autoHide", value: function (u) {
                                var f = this, O = parseInt(this.getClassProperty(u, "--auto-hide", "0"));
                                O && (u.autoHide = setTimeout(function () {
                                    f.close(u)
                                }, O))
                            }
                        }, {
                            key: "_checkTimer", value: function (u) {
                                u.autoHide && (clearTimeout(u.autoHide), delete u.autoHide)
                            }
                        }, {
                            key: "_onBackdropClick", value: function (u) {
                                this.getClassProperty(u, "--overlay-backdrop", "true") !== "static" && this.close(u)
                            }
                        }, {
                            key: "_buildBackdrop", value: function (u) {
                                var f, O = this, C = u.getAttribute("data-hs-overlay-backdrop-container") || !1,
                                    S = document.createElement("div"),
                                    j = "transition duration fixed inset-0 z-50 bg-gray-900 bg-opacity-50 dark:bg-opacity-80 hs-overlay-backdrop",
                                    R = function (N, ie) {
                                        var re = typeof Symbol < "u" && N[Symbol.iterator] || N["@@iterator"];
                                        if (!re) {
                                            if (Array.isArray(N) || (re = function (me, it) {
                                                if (me) {
                                                    if (typeof me == "string") return d(me, it);
                                                    var ze = Object.prototype.toString.call(me).slice(8, -1);
                                                    return ze === "Object" && me.constructor && (ze = me.constructor.name), ze === "Map" || ze === "Set" ? Array.from(me) : ze === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(ze) ? d(me, it) : void 0
                                                }
                                            }(N)) || ie && N && typeof N.length == "number") {
                                                re && (N = re);
                                                var Ye = 0, Ae = function () {
                                                };
                                                return {
                                                    s: Ae, n: function () {
                                                        return Ye >= N.length ? {done: !0} : {done: !1, value: N[Ye++]}
                                                    }, e: function (me) {
                                                        throw me
                                                    }, f: Ae
                                                }
                                            }
                                            throw new TypeError(`迭代不可迭代实例的尝试无效.
为了可迭代，非数组对象必须具有 [Symbol.iterator]() 方法.`)
                                        }
                                        var De, Ve = !0, lt = !1;
                                        return {
                                            s: function () {
                                                re = re.call(N)
                                            }, n: function () {
                                                var me = re.next();
                                                return Ve = me.done, me
                                            }, e: function (me) {
                                                lt = !0, De = me
                                            }, f: function () {
                                                try {
                                                    Ve || re.return == null || re.return()
                                                } finally {
                                                    if (lt) throw De
                                                }
                                            }
                                        }
                                    }(u.classList.values());
                                try {
                                    for (R.s(); !(f = R.n()).done;) {
                                        var q = f.value;
                                        q.startsWith("hs-overlay-backdrop-open:") && (j += " ".concat(q))
                                    }
                                } catch (N) {
                                    R.e(N)
                                } finally {
                                    R.f()
                                }
                                var B = this.getClassProperty(u, "--overlay-backdrop", "true") !== "static";
                                this.getClassProperty(u, "--overlay-backdrop", "true") === "false" || (C && ((S = document.querySelector(C).cloneNode(!0)).classList.remove("hidden"), j = S.classList, S.classList = ""), B && S.addEventListener("click", function () {
                                    return O.close(u)
                                }, !0), S.setAttribute("data-hs-overlay-backdrop-template", ""), document.body.appendChild(S), setTimeout(function () {
                                    S.classList = j
                                }))
                            }
                        }, {
                            key: "_destroyBackdrop", value: function () {
                                var u = document.querySelector("[data-hs-overlay-backdrop-template]");
                                u && (this.openNextOverlay && (u.style.transitionDuration = "".concat(1.8 * parseFloat(window.getComputedStyle(u).transitionDuration.replace(/[^\d.-]/g, "")), "s")), u.classList.add("opacity-0"), this.afterTransition(u, function () {
                                    u.remove()
                                }))
                            }
                        }, {
                            key: "_focusInput", value: function (u) {
                                var f = u.querySelector("[autofocus]");
                                f && f.focus()
                            }
                        }]) && g(h.prototype, p), Object.defineProperty(h, "prototype", {writable: !1}), T
                    }(c(765).Z);
                    window.HSOverlay = new L, document.addEventListener("load", window.HSOverlay.init())
                }, 181: (s, l, c) => {
                    function a(L) {
                        return a = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function (m) {
                            return typeof m
                        } : function (m) {
                            return m && typeof Symbol == "function" && m.constructor === Symbol && m !== Symbol.prototype ? "symbol" : typeof m
                        }, a(L)
                    }

                    function d(L, m) {
                        for (var h = 0; h < m.length; h++) {
                            var p = m[h];
                            p.enumerable = p.enumerable || !1, p.configurable = !0, "value" in p && (p.writable = !0), Object.defineProperty(L, p.key, p)
                        }
                    }

                    function g(L, m) {
                        return g = Object.setPrototypeOf || function (h, p) {
                            return h.__proto__ = p, h
                        }, g(L, m)
                    }

                    function k(L, m) {
                        if (m && (a(m) === "object" || typeof m == "function")) return m;
                        if (m !== void 0) throw new TypeError("派生构造函数只能返回对象或未定义");
                        return function (h) {
                            if (h === void 0) throw new ReferenceError("这还没有被初始化——super()还没有被调用");
                            return h
                        }(L)
                    }

                    function x(L) {
                        return x = Object.setPrototypeOf ? Object.getPrototypeOf : function (m) {
                            return m.__proto__ || Object.getPrototypeOf(m)
                        }, x(L)
                    }

                    var D = function (L) {
                        (function (T, u) {
                            if (typeof u != "function" && u !== null) throw new TypeError("超级表达式必须为空或者为函数");
                            T.prototype = Object.create(u && u.prototype, {
                                constructor: {
                                    value: T,
                                    writable: !0,
                                    configurable: !0
                                }
                            }), Object.defineProperty(T, "prototype", {writable: !1}), u && g(T, u)
                        })(P, L);
                        var m, h, p, _, A = (p = P, _ = function () {
                            if (typeof Reflect > "u" || !Reflect.construct || Reflect.construct.sham) return !1;
                            if (typeof Proxy == "function") return !0;
                            try {
                                return Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {
                                })), !0
                            } catch {
                                return !1
                            }
                        }(), function () {
                            var T, u = x(p);
                            if (_) {
                                var f = x(this).constructor;
                                T = Reflect.construct(u, arguments, f)
                            } else T = u.apply(this, arguments);
                            return k(this, T)
                        });

                        function P() {
                            return function (T, u) {
                                if (!(T instanceof u)) throw new TypeError("无法将类作为函数调用")
                            }(this, P), A.call(this, "[data-hs-remove-element]")
                        }

                        return m = P, (h = [{
                            key: "init", value: function () {
                                var T = this;
                                document.addEventListener("click", function (u) {
                                    var f = u.target.closest(T.selector);
                                    if (f) {
                                        var O = document.querySelector(f.getAttribute("data-hs-remove-element"));
                                        O && (O.classList.add("hs-removing"), T.afterTransition(O, function () {
                                            O.remove()
                                        }))
                                    }
                                })
                            }
                        }]) && d(m.prototype, h), Object.defineProperty(m, "prototype", {writable: !1}), P
                    }(c(765).Z);
                    window.HSRemoveElement = new D, document.addEventListener("load", window.HSRemoveElement.init())
                }, 778: (s, l, c) => {
                    function a(L) {
                        return a = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function (m) {
                            return typeof m
                        } : function (m) {
                            return m && typeof Symbol == "function" && m.constructor === Symbol && m !== Symbol.prototype ? "symbol" : typeof m
                        }, a(L)
                    }

                    function d(L, m) {
                        for (var h = 0; h < m.length; h++) {
                            var p = m[h];
                            p.enumerable = p.enumerable || !1, p.configurable = !0, "value" in p && (p.writable = !0), Object.defineProperty(L, p.key, p)
                        }
                    }

                    function g(L, m) {
                        return g = Object.setPrototypeOf || function (h, p) {
                            return h.__proto__ = p, h
                        }, g(L, m)
                    }

                    function k(L, m) {
                        if (m && (a(m) === "object" || typeof m == "function")) return m;
                        if (m !== void 0) throw new TypeError("派生构造函数只能返回对象或未定义");
                        return function (h) {
                            if (h === void 0) throw new ReferenceError("这还没有被初始化——super()还没有被调用");
                            return h
                        }(L)
                    }

                    function x(L) {
                        return x = Object.setPrototypeOf ? Object.getPrototypeOf : function (m) {
                            return m.__proto__ || Object.getPrototypeOf(m)
                        }, x(L)
                    }

                    var D = function (L) {
                        (function (T, u) {
                            if (typeof u != "function" && u !== null) throw new TypeError("超级表达式必须为空或者为函数");
                            T.prototype = Object.create(u && u.prototype, {
                                constructor: {
                                    value: T,
                                    writable: !0,
                                    configurable: !0
                                }
                            }), Object.defineProperty(T, "prototype", {writable: !1}), u && g(T, u)
                        })(P, L);
                        var m, h, p, _, A = (p = P, _ = function () {
                            if (typeof Reflect > "u" || !Reflect.construct || Reflect.construct.sham) return !1;
                            if (typeof Proxy == "function") return !0;
                            try {
                                return Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {
                                })), !0
                            } catch {
                                return !1
                            }
                        }(), function () {
                            var T, u = x(p);
                            if (_) {
                                var f = x(this).constructor;
                                T = Reflect.construct(u, arguments, f)
                            } else T = u.apply(this, arguments);
                            return k(this, T)
                        });

                        function P() {
                            var T;
                            return function (u, f) {
                                if (!(u instanceof f)) throw new TypeError("无法将类作为函数调用")
                            }(this, P), (T = A.call(this, "[data-hs-scrollspy] ")).activeSection = null, T
                        }

                        return m = P, (h = [{
                            key: "init", value: function () {
                                var T = this;
                                document.querySelectorAll(this.selector).forEach(function (u) {
                                    var f = document.querySelector(u.getAttribute("data-hs-scrollspy")),
                                        O = u.querySelectorAll("[href]"), C = f.children,
                                        S = u.getAttribute("data-hs-scrollspy-scrollable-parent") ? document.querySelector(u.getAttribute("data-hs-scrollspy-scrollable-parent")) : document;
                                    Array.from(C).forEach(function (j) {
                                        j.getAttribute("id") && S.addEventListener("scroll", function (R) {
                                            return T._update({
                                                $scrollspyEl: u,
                                                $scrollspyContentEl: f,
                                                links: O,
                                                $sectionEl: j,
                                                sections: C,
                                                ev: R
                                            })
                                        })
                                    }), O.forEach(function (j) {
                                        j.addEventListener("click", function (R) {
                                            R.preventDefault(), j.getAttribute("href") !== "javascript:;" && T._scrollTo({
                                                $scrollspyEl: u,
                                                $scrollableEl: S,
                                                $link: j
                                            })
                                        })
                                    })
                                })
                            }
                        }, {
                            key: "_update", value: function (T) {
                                var u = T.ev, f = T.$scrollspyEl, O = (T.sections, T.links), C = T.$sectionEl,
                                    S = parseInt(this.getClassProperty(f, "--scrollspy-offset", "0")),
                                    j = this.getClassProperty(C, "--scrollspy-offset") || S,
                                    R = u.target === document ? 0 : parseInt(u.target.getBoundingClientRect().top),
                                    q = parseInt(C.getBoundingClientRect().top) - j - R, B = C.offsetHeight;
                                if (q <= 0 && q + B > 0) {
                                    if (this.activeSection === C) return;
                                    O.forEach(function (Ye) {
                                        Ye.classList.remove("active")
                                    });
                                    var N = f.querySelector('[href="#'.concat(C.getAttribute("id"), '"]'));
                                    if (N) {
                                        N.classList.add("active");
                                        var ie = N.closest("[data-hs-scrollspy-group]");
                                        if (ie) {
                                            var re = ie.querySelector("[href]");
                                            re && re.classList.add("active")
                                        }
                                    }
                                    this.activeSection = C
                                }
                            }
                        }, {
                            key: "_scrollTo", value: function (T) {
                                var u = T.$scrollspyEl, f = T.$scrollableEl, O = T.$link,
                                    C = document.querySelector(O.getAttribute("href")),
                                    S = parseInt(this.getClassProperty(u, "--scrollspy-offset", "0")),
                                    j = this.getClassProperty(C, "--scrollspy-offset") || S,
                                    R = f === document ? 0 : f.offsetTop, q = C.offsetTop - j - R,
                                    B = f === document ? window : f;
                                this._fireEvent("scroll", u), this._dispatch("scroll.hs.scrollspy", u, u), window.history.replaceState(null, null, O.getAttribute("href")), B.scrollTo({
                                    top: q,
                                    left: 0,
                                    behavior: "smooth"
                                })
                            }
                        }]) && d(m.prototype, h), Object.defineProperty(m, "prototype", {writable: !1}), P
                    }(c(765).Z);
                    window.HSScrollspy = new D, document.addEventListener("load", window.HSScrollspy.init())
                }, 51: (s, l, c) => {
                    function a(h) {
                        return a = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function (p) {
                            return typeof p
                        } : function (p) {
                            return p && typeof Symbol == "function" && p.constructor === Symbol && p !== Symbol.prototype ? "symbol" : typeof p
                        }, a(h)
                    }

                    function d(h) {
                        return function (p) {
                            if (Array.isArray(p)) return g(p)
                        }(h) || function (p) {
                            if (typeof Symbol < "u" && p[Symbol.iterator] != null || p["@@iterator"] != null) return Array.from(p)
                        }(h) || function (p, _) {
                            if (p) {
                                if (typeof p == "string") return g(p, _);
                                var A = Object.prototype.toString.call(p).slice(8, -1);
                                return A === "Object" && p.constructor && (A = p.constructor.name), A === "Map" || A === "Set" ? Array.from(p) : A === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(A) ? g(p, _) : void 0
                            }
                        }(h) || function () {
                            throw new TypeError(`传播不可迭代实例的尝试无效.
为了可迭代，非数组对象必须具有 [Symbol.iterator]() 方法.`)
                        }()
                    }

                    function g(h, p) {
                        (p == null || p > h.length) && (p = h.length);
                        for (var _ = 0, A = new Array(p); _ < p; _++) A[_] = h[_];
                        return A
                    }

                    function k(h, p) {
                        for (var _ = 0; _ < p.length; _++) {
                            var A = p[_];
                            A.enumerable = A.enumerable || !1, A.configurable = !0, "value" in A && (A.writable = !0), Object.defineProperty(h, A.key, A)
                        }
                    }

                    function x(h, p) {
                        return x = Object.setPrototypeOf || function (_, A) {
                            return _.__proto__ = A, _
                        }, x(h, p)
                    }

                    function D(h, p) {
                        if (p && (a(p) === "object" || typeof p == "function")) return p;
                        if (p !== void 0) throw new TypeError("派生构造函数只能返回对象或未定义");
                        return function (_) {
                            if (_ === void 0) throw new ReferenceError("这还没有被初始化——super()还没有被调用");
                            return _
                        }(h)
                    }

                    function L(h) {
                        return L = Object.setPrototypeOf ? Object.getPrototypeOf : function (p) {
                            return p.__proto__ || Object.getPrototypeOf(p)
                        }, L(h)
                    }

                    var m = function (h) {
                        (function (f, O) {
                            if (typeof O != "function" && O !== null) throw new TypeError("超级表达式必须为空或者为函数");
                            f.prototype = Object.create(O && O.prototype, {
                                constructor: {
                                    value: f,
                                    writable: !0,
                                    configurable: !0
                                }
                            }), Object.defineProperty(f, "prototype", {writable: !1}), O && x(f, O)
                        })(u, h);
                        var p, _, A, P, T = (A = u, P = function () {
                            if (typeof Reflect > "u" || !Reflect.construct || Reflect.construct.sham) return !1;
                            if (typeof Proxy == "function") return !0;
                            try {
                                return Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {
                                })), !0
                            } catch {
                                return !1
                            }
                        }(), function () {
                            var f, O = L(A);
                            if (P) {
                                var C = L(this).constructor;
                                f = Reflect.construct(O, arguments, C)
                            } else f = O.apply(this, arguments);
                            return D(this, f)
                        });

                        function u() {
                            return function (f, O) {
                                if (!(f instanceof O)) throw new TypeError("无法将类作为函数调用")
                            }(this, u), T.call(this, "[data-hs-tab]")
                        }

                        return p = u, (_ = [{
                            key: "init", value: function () {
                                var f = this;
                                document.addEventListener("keydown", this._keyboardSupport.bind(this)), document.addEventListener("click", function (O) {
                                    var C = O.target.closest(f.selector);
                                    C && f.open(C)
                                }), document.querySelectorAll("[hs-data-tab-select]").forEach(function (O) {
                                    var C = document.querySelector(O.getAttribute("hs-data-tab-select"));
                                    C && C.addEventListener("change", function (S) {
                                        var j = document.querySelector('[data-hs-tab="'.concat(S.target.value, '"]'));
                                        j && f.open(j)
                                    })
                                })
                            }
                        }, {
                            key: "open", value: function (f) {
                                var O = document.querySelector(f.getAttribute("data-hs-tab")),
                                    C = d(f.parentElement.children), S = d(O.parentElement.children),
                                    j = f.closest("[hs-data-tab-select]"),
                                    R = j ? document.querySelector(j.getAttribute("data-hs-tab")) : null;
                                C.forEach(function (q) {
                                    return q.classList.remove("active")
                                }), S.forEach(function (q) {
                                    return q.classList.add("hidden")
                                }), f.classList.add("active"), O.classList.remove("hidden"), this._fireEvent("change", f), this._dispatch("change.hs.tab", f, f), R && (R.value = f.getAttribute("data-hs-tab"))
                            }
                        }, {
                            key: "_keyboardSupport", value: function (f) {
                                var O = f.target.closest(this.selector);
                                if (O) {
                                    var C = O.closest('[role="tablist"]').getAttribute("data-hs-tabs-vertical") === "true";
                                    return (C ? f.keyCode === 38 : f.keyCode === 37) ? (f.preventDefault(), this._left(O)) : (C ? f.keyCode === 40 : f.keyCode === 39) ? (f.preventDefault(), this._right(O)) : f.keyCode === 36 ? (f.preventDefault(), this._start(O)) : f.keyCode === 35 ? (f.preventDefault(), this._end(O)) : void 0
                                }
                            }
                        }, {
                            key: "_right", value: function (f) {
                                var O = f.closest('[role="tablist"]');
                                if (O) {
                                    var C = d(O.querySelectorAll(this.selector)).filter(function (R) {
                                        return !R.disabled
                                    }), S = O.querySelector("button:focus"), j = C.findIndex(function (R) {
                                        return R === S
                                    });
                                    j + 1 < C.length ? j++ : j = 0, C[j].focus(), this.open(C[j])
                                }
                            }
                        }, {
                            key: "_left", value: function (f) {
                                var O = f.closest('[role="tablist"]');
                                if (O) {
                                    var C = d(O.querySelectorAll(this.selector)).filter(function (R) {
                                        return !R.disabled
                                    }).reverse(), S = O.querySelector("button:focus"), j = C.findIndex(function (R) {
                                        return R === S
                                    });
                                    j + 1 < C.length ? j++ : j = 0, C[j].focus(), this.open(C[j])
                                }
                            }
                        }, {
                            key: "_start", value: function (f) {
                                var O = f.closest('[role="tablist"]');
                                if (O) {
                                    var C = d(O.querySelectorAll(this.selector)).filter(function (S) {
                                        return !S.disabled
                                    });
                                    C.length && (C[0].focus(), this.open(C[0]))
                                }
                            }
                        }, {
                            key: "_end", value: function (f) {
                                var O = f.closest('[role="tablist"]');
                                if (O) {
                                    var C = d(O.querySelectorAll(this.selector)).reverse().filter(function (S) {
                                        return !S.disabled
                                    });
                                    C.length && (C[0].focus(), this.open(C[0]))
                                }
                            }
                        }]) && k(p.prototype, _), Object.defineProperty(p, "prototype", {writable: !1}), u
                    }(c(765).Z);
                    window.HSTabs = new m, document.addEventListener("load", window.HSTabs.init())
                }, 185: (s, l, c) => {
                    var a = c(765), d = c(714);

                    function g(h) {
                        return g = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function (p) {
                            return typeof p
                        } : function (p) {
                            return p && typeof Symbol == "function" && p.constructor === Symbol && p !== Symbol.prototype ? "symbol" : typeof p
                        }, g(h)
                    }

                    function k(h, p) {
                        for (var _ = 0; _ < p.length; _++) {
                            var A = p[_];
                            A.enumerable = A.enumerable || !1, A.configurable = !0, "value" in A && (A.writable = !0), Object.defineProperty(h, A.key, A)
                        }
                    }

                    function x(h, p) {
                        return x = Object.setPrototypeOf || function (_, A) {
                            return _.__proto__ = A, _
                        }, x(h, p)
                    }

                    function D(h, p) {
                        if (p && (g(p) === "object" || typeof p == "function")) return p;
                        if (p !== void 0) throw new TypeError("派生构造函数只能返回对象或未定义");
                        return function (_) {
                            if (_ === void 0) throw new ReferenceError("这还没有被初始化——super()还没有被调用");
                            return _
                        }(h)
                    }

                    function L(h) {
                        return L = Object.setPrototypeOf ? Object.getPrototypeOf : function (p) {
                            return p.__proto__ || Object.getPrototypeOf(p)
                        }, L(h)
                    }

                    var m = function (h) {
                        (function (f, O) {
                            if (typeof O != "function" && O !== null) throw new TypeError("超级表达式必须为空或者为函数");
                            f.prototype = Object.create(O && O.prototype, {
                                constructor: {
                                    value: f,
                                    writable: !0,
                                    configurable: !0
                                }
                            }), Object.defineProperty(f, "prototype", {writable: !1}), O && x(f, O)
                        })(u, h);
                        var p, _, A, P, T = (A = u, P = function () {
                            if (typeof Reflect > "u" || !Reflect.construct || Reflect.construct.sham) return !1;
                            if (typeof Proxy == "function") return !0;
                            try {
                                return Boolean.prototype.valueOf.call(Reflect.construct(Boolean, [], function () {
                                })), !0
                            } catch {
                                return !1
                            }
                        }(), function () {
                            var f, O = L(A);
                            if (P) {
                                var C = L(this).constructor;
                                f = Reflect.construct(O, arguments, C)
                            } else f = O.apply(this, arguments);
                            return D(this, f)
                        });

                        function u() {
                            return function (f, O) {
                                if (!(f instanceof O)) throw new TypeError("无法将类作为函数调用")
                            }(this, u), T.call(this, ".hs-tooltip")
                        }

                        return p = u, (_ = [{
                            key: "init", value: function () {
                                var f = this;
                                document.addEventListener("click", function (O) {
                                    var C = O.target.closest(f.selector);
                                    C && f.getClassProperty(C, "--trigger") === "focus" && f._focus(C), C && f.getClassProperty(C, "--trigger") === "click" && f._click(C)
                                }), document.addEventListener("mousemove", function (O) {
                                    var C = O.target.closest(f.selector);
                                    C && f.getClassProperty(C, "--trigger") !== "focus" && f.getClassProperty(C, "--trigger") !== "click" && f._hover(C)
                                })
                            }
                        }, {
                            key: "_hover", value: function (f) {
                                var O = this;
                                if (!f.classList.contains("show")) {
                                    var C = f.querySelector(".hs-tooltip-toggle"),
                                        S = f.querySelector(".hs-tooltip-content"),
                                        j = this.getClassProperty(f, "--placement");
                                    (0, d.fi)(C, S, {
                                        placement: j || "top",
                                        strategy: "fixed",
                                        modifiers: [{name: "offset", options: {offset: [0, 5]}}]
                                    }), this.show(f), f.addEventListener("mouseleave", function R(q) {
                                        q.relatedTarget.closest(O.selector) && q.relatedTarget.closest(O.selector) == f || (O.hide(f), f.removeEventListener("mouseleave", R, !0))
                                    }, !0)
                                }
                            }
                        }, {
                            key: "_focus", value: function (f) {
                                var O = this, C = f.querySelector(".hs-tooltip-toggle"),
                                    S = f.querySelector(".hs-tooltip-content"),
                                    j = this.getClassProperty(f, "--placement"),
                                    R = this.getClassProperty(f, "--strategy");
                                (0, d.fi)(C, S, {
                                    placement: j || "top",
                                    strategy: R || "fixed",
                                    modifiers: [{name: "offset", options: {offset: [0, 5]}}]
                                }), this.show(f), f.addEventListener("blur", function q() {
                                    O.hide(f), f.removeEventListener("blur", q, !0)
                                }, !0)
                            }
                        }, {
                            key: "_click", value: function (f) {
                                var O = this;
                                if (!f.classList.contains("show")) {
                                    var C = f.querySelector(".hs-tooltip-toggle"),
                                        S = f.querySelector(".hs-tooltip-content"),
                                        j = this.getClassProperty(f, "--placement"),
                                        R = this.getClassProperty(f, "--strategy");
                                    (0, d.fi)(C, S, {
                                        placement: j || "top",
                                        strategy: R || "fixed",
                                        modifiers: [{name: "offset", options: {offset: [0, 5]}}]
                                    }), this.show(f);
                                    var q = function B(N) {
                                        setTimeout(function () {
                                            O.hide(f), f.removeEventListener("click", B, !0), f.removeEventListener("blur", B, !0)
                                        })
                                    };
                                    f.addEventListener("blur", q, !0), f.addEventListener("click", q, !0)
                                }
                            }
                        }, {
                            key: "show", value: function (f) {
                                var O = this;
                                f.querySelector(".hs-tooltip-content").classList.remove("hidden"), setTimeout(function () {
                                    f.classList.add("show"), O._fireEvent("show", f), O._dispatch("show.hs.tooltip", f, f)
                                })
                            }
                        }, {
                            key: "hide", value: function (f) {
                                var O = f.querySelector(".hs-tooltip-content");
                                f.classList.remove("show"), this._fireEvent("hide", f), this._dispatch("hide.hs.tooltip", f, f), this.afterTransition(O, function () {
                                    f.classList.contains("show") || O.classList.add("hidden")
                                })
                            }
                        }]) && k(p.prototype, _), Object.defineProperty(p, "prototype", {writable: !1}), u
                    }(a.Z);
                    window.HSTooltip = new m, document.addEventListener("load", window.HSTooltip.init())
                }, 765: (s, l, c) => {
                    function a(g, k) {
                        for (var x = 0; x < k.length; x++) {
                            var D = k[x];
                            D.enumerable = D.enumerable || !1, D.configurable = !0, "value" in D && (D.writable = !0), Object.defineProperty(g, D.key, D)
                        }
                    }

                    c.d(l, {Z: () => d});
                    var d = function () {
                        function g(D, L) {
                            (function (m, h) {
                                if (!(m instanceof h)) throw new TypeError("无法将类作为函数调用")
                            })(this, g), this.$collection = [], this.selector = D, this.config = L, this.events = {}
                        }

                        var k, x;
                        return k = g, x = [{
                            key: "_fireEvent", value: function (D) {
                                var L = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : null;
                                this.events.hasOwnProperty(D) && this.events[D](L)
                            }
                        }, {
                            key: "_dispatch", value: function (D, L) {
                                var m = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : null,
                                    h = new CustomEvent(D, {
                                        detail: {payload: m},
                                        bubbles: !0,
                                        cancelable: !0,
                                        composed: !1
                                    });
                                L.dispatchEvent(h)
                            }
                        }, {
                            key: "on", value: function (D, L) {
                                this.events[D] = L
                            }
                        }, {
                            key: "afterTransition", value: function (D, L) {
                                window.getComputedStyle(D, null).getPropertyValue("transition") !== "all 0s ease 0s" ? D.addEventListener("transitionend", function m() {
                                    L(), D.removeEventListener("transitionend", m, !0)
                                }, !0) : L()
                            }
                        }, {
                            key: "getClassProperty", value: function (D, L) {
                                var m = arguments.length > 2 && arguments[2] !== void 0 ? arguments[2] : "",
                                    h = (window.getComputedStyle(D).getPropertyValue(L) || m).replace(" ", "");
                                return h
                            }
                        }], x && a(k.prototype, x), Object.defineProperty(k, "prototype", {writable: !1}), g
                    }()
                }, 714: (s, l, c) => {
                    function a(b) {
                        if (b == null) return window;
                        if (b.toString() !== "[object Window]") {
                            var y = b.ownerDocument;
                            return y && y.defaultView || window
                        }
                        return b
                    }

                    function d(b) {
                        return b instanceof a(b).Element || b instanceof Element
                    }

                    function g(b) {
                        return b instanceof a(b).HTMLElement || b instanceof HTMLElement
                    }

                    function k(b) {
                        return typeof ShadowRoot < "u" && (b instanceof a(b).ShadowRoot || b instanceof ShadowRoot)
                    }

                    c.d(l, {fi: () => gr});
                    var x = Math.max, D = Math.min, L = Math.round;

                    function m(b, y) {
                        y === void 0 && (y = !1);
                        var $ = b.getBoundingClientRect(), H = 1, z = 1;
                        if (g(b) && y) {
                            var Q = b.offsetHeight, V = b.offsetWidth;
                            V > 0 && (H = L($.width) / V || 1), Q > 0 && (z = L($.height) / Q || 1)
                        }
                        return {
                            width: $.width / H,
                            height: $.height / z,
                            top: $.top / z,
                            right: $.right / H,
                            bottom: $.bottom / z,
                            left: $.left / H,
                            x: $.left / H,
                            y: $.top / z
                        }
                    }

                    function h(b) {
                        var y = a(b);
                        return {scrollLeft: y.pageXOffset, scrollTop: y.pageYOffset}
                    }

                    function p(b) {
                        return b ? (b.nodeName || "").toLowerCase() : null
                    }

                    function _(b) {
                        return ((d(b) ? b.ownerDocument : b.document) || window.document).documentElement
                    }

                    function A(b) {
                        return m(_(b)).left + h(b).scrollLeft
                    }

                    function P(b) {
                        return a(b).getComputedStyle(b)
                    }

                    function T(b) {
                        var y = P(b), $ = y.overflow, H = y.overflowX, z = y.overflowY;
                        return /auto|scroll|overlay|hidden/.test($ + z + H)
                    }

                    function u(b, y, $) {
                        $ === void 0 && ($ = !1);
                        var H, z, Q = g(y), V = g(y) && function (ee) {
                            var Ee = ee.getBoundingClientRect(), ae = L(Ee.width) / ee.offsetWidth || 1,
                                he = L(Ee.height) / ee.offsetHeight || 1;
                            return ae !== 1 || he !== 1
                        }(y), W = _(y), K = m(b, V), te = {scrollLeft: 0, scrollTop: 0}, ne = {x: 0, y: 0};
                        return (Q || !Q && !$) && ((p(y) !== "body" || T(W)) && (te = (H = y) !== a(H) && g(H) ? {
                            scrollLeft: (z = H).scrollLeft,
                            scrollTop: z.scrollTop
                        } : h(H)), g(y) ? ((ne = m(y, !0)).x += y.clientLeft, ne.y += y.clientTop) : W && (ne.x = A(W))), {
                            x: K.left + te.scrollLeft - ne.x,
                            y: K.top + te.scrollTop - ne.y,
                            width: K.width,
                            height: K.height
                        }
                    }

                    function f(b) {
                        var y = m(b), $ = b.offsetWidth, H = b.offsetHeight;
                        return Math.abs(y.width - $) <= 1 && ($ = y.width), Math.abs(y.height - H) <= 1 && (H = y.height), {
                            x: b.offsetLeft,
                            y: b.offsetTop,
                            width: $,
                            height: H
                        }
                    }

                    function O(b) {
                        return p(b) === "html" ? b : b.assignedSlot || b.parentNode || (k(b) ? b.host : null) || _(b)
                    }

                    function C(b) {
                        return ["html", "body", "#document"].indexOf(p(b)) >= 0 ? b.ownerDocument.body : g(b) && T(b) ? b : C(O(b))
                    }

                    function S(b, y) {
                        var $;
                        y === void 0 && (y = []);
                        var H = C(b), z = H === (($ = b.ownerDocument) == null ? void 0 : $.body), Q = a(H),
                            V = z ? [Q].concat(Q.visualViewport || [], T(H) ? H : []) : H, W = y.concat(V);
                        return z ? W : W.concat(S(O(V)))
                    }

                    function j(b) {
                        return ["table", "td", "th"].indexOf(p(b)) >= 0
                    }

                    function R(b) {
                        return g(b) && P(b).position !== "fixed" ? b.offsetParent : null
                    }

                    function q(b) {
                        for (var y = a(b), $ = R(b); $ && j($) && P($).position === "static";) $ = R($);
                        return $ && (p($) === "html" || p($) === "body" && P($).position === "static") ? y : $ || function (H) {
                            var z = navigator.userAgent.toLowerCase().indexOf("firefox") !== -1;
                            if (navigator.userAgent.indexOf("Trident") !== -1 && g(H) && P(H).position === "fixed") return null;
                            for (var Q = O(H); g(Q) && ["html", "body"].indexOf(p(Q)) < 0;) {
                                var V = P(Q);
                                if (V.transform !== "none" || V.perspective !== "none" || V.contain === "paint" || ["transform", "perspective"].indexOf(V.willChange) !== -1 || z && V.willChange === "filter" || z && V.filter && V.filter !== "none") return Q;
                                Q = Q.parentNode
                            }
                            return null
                        }(b) || y
                    }

                    var B = "top", N = "bottom", ie = "right", re = "left", Ye = "auto", Ae = [B, N, ie, re],
                        De = "start", Ve = "end", lt = "viewport", me = "popper", it = Ae.reduce(function (b, y) {
                            return b.concat([y + "-" + De, y + "-" + Ve])
                        }, []), ze = [].concat(Ae, [Ye]).reduce(function (b, y) {
                            return b.concat([y, y + "-" + De, y + "-" + Ve])
                        }, []),
                        at = ["beforeRead", "read", "afterRead", "beforeMain", "main", "afterMain", "beforeWrite", "write", "afterWrite"];

                    function Lt(b) {
                        var y = new Map, $ = new Set, H = [];

                        function z(Q) {
                            $.add(Q.name), [].concat(Q.requires || [], Q.requiresIfExists || []).forEach(function (V) {
                                if (!$.has(V)) {
                                    var W = y.get(V);
                                    W && z(W)
                                }
                            }), H.push(Q)
                        }

                        return b.forEach(function (Q) {
                            y.set(Q.name, Q)
                        }), b.forEach(function (Q) {
                            $.has(Q.name) || z(Q)
                        }), H
                    }

                    var Tt = {placement: "bottom", modifiers: [], strategy: "absolute"};

                    function Te() {
                        for (var b = arguments.length, y = new Array(b), $ = 0; $ < b; $++) y[$] = arguments[$];
                        return !y.some(function (H) {
                            return !(H && typeof H.getBoundingClientRect == "function")
                        })
                    }

                    function ct(b) {
                        b === void 0 && (b = {});
                        var y = b, $ = y.defaultModifiers, H = $ === void 0 ? [] : $, z = y.defaultOptions,
                            Q = z === void 0 ? Tt : z;
                        return function (V, W, K) {
                            K === void 0 && (K = Q);
                            var te, ne, ee = {
                                placement: "bottom",
                                orderedModifiers: [],
                                options: Object.assign({}, Tt, Q),
                                modifiersData: {},
                                elements: {reference: V, popper: W},
                                attributes: {},
                                styles: {}
                            }, Ee = [], ae = !1, he = {
                                state: ee, setOptions: function (le) {
                                    var Ce = typeof le == "function" ? le(ee.options) : le;
                                    pe(), ee.options = Object.assign({}, Q, ee.options, Ce), ee.scrollParents = {
                                        reference: d(V) ? S(V) : V.contextElement ? S(V.contextElement) : [],
                                        popper: S(W)
                                    };
                                    var ve, fe, ye = function (ce) {
                                        var ue = Lt(ce);
                                        return at.reduce(function (de, ge) {
                                            return de.concat(ue.filter(function (be) {
                                                return be.phase === ge
                                            }))
                                        }, [])
                                    }((ve = [].concat(H, ee.options.modifiers), fe = ve.reduce(function (ce, ue) {
                                        var de = ce[ue.name];
                                        return ce[ue.name] = de ? Object.assign({}, de, ue, {
                                            options: Object.assign({}, de.options, ue.options),
                                            data: Object.assign({}, de.data, ue.data)
                                        }) : ue, ce
                                    }, {}), Object.keys(fe).map(function (ce) {
                                        return fe[ce]
                                    })));
                                    return ee.orderedModifiers = ye.filter(function (ce) {
                                        return ce.enabled
                                    }), ee.orderedModifiers.forEach(function (ce) {
                                        var ue = ce.name, de = ce.options, ge = de === void 0 ? {} : de, be = ce.effect;
                                        if (typeof be == "function") {
                                            var Me = be({state: ee, name: ue, instance: he, options: ge});
                                            Ee.push(Me || function () {
                                            })
                                        }
                                    }), he.update()
                                }, forceUpdate: function () {
                                    if (!ae) {
                                        var le = ee.elements, Ce = le.reference, ve = le.popper;
                                        if (Te(Ce, ve)) {
                                            ee.rects = {
                                                reference: u(Ce, q(ve), ee.options.strategy === "fixed"),
                                                popper: f(ve)
                                            }, ee.reset = !1, ee.placement = ee.options.placement, ee.orderedModifiers.forEach(function (be) {
                                                return ee.modifiersData[be.name] = Object.assign({}, be.data)
                                            });
                                            for (var fe = 0; fe < ee.orderedModifiers.length; fe++) if (ee.reset !== !0) {
                                                var ye = ee.orderedModifiers[fe], ce = ye.fn, ue = ye.options,
                                                    de = ue === void 0 ? {} : ue, ge = ye.name;
                                                typeof ce == "function" && (ee = ce({
                                                    state: ee,
                                                    options: de,
                                                    name: ge,
                                                    instance: he
                                                }) || ee)
                                            } else ee.reset = !1, fe = -1
                                        }
                                    }
                                }, update: (te = function () {
                                    return new Promise(function (le) {
                                        he.forceUpdate(), le(ee)
                                    })
                                }, function () {
                                    return ne || (ne = new Promise(function (le) {
                                        Promise.resolve().then(function () {
                                            ne = void 0, le(te())
                                        })
                                    })), ne
                                }), destroy: function () {
                                    pe(), ae = !0
                                }
                            };
                            if (!Te(V, W)) return he;

                            function pe() {
                                Ee.forEach(function (le) {
                                    return le()
                                }), Ee = []
                            }

                            return he.setOptions(K).then(function (le) {
                                !ae && K.onFirstUpdate && K.onFirstUpdate(le)
                            }), he
                        }
                    }

                    var Pe = {passive: !0};

                    function Be(b) {
                        return b.split("-")[0]
                    }

                    function Oe(b) {
                        return b.split("-")[1]
                    }

                    function Le(b) {
                        return ["top", "bottom"].indexOf(b) >= 0 ? "x" : "y"
                    }

                    function X(b) {
                        var y, $ = b.reference, H = b.element, z = b.placement, Q = z ? Be(z) : null,
                            V = z ? Oe(z) : null, W = $.x + $.width / 2 - H.width / 2,
                            K = $.y + $.height / 2 - H.height / 2;
                        switch (Q) {
                            case B:
                                y = {x: W, y: $.y - H.height};
                                break;
                            case N:
                                y = {x: W, y: $.y + $.height};
                                break;
                            case ie:
                                y = {x: $.x + $.width, y: K};
                                break;
                            case re:
                                y = {x: $.x - H.width, y: K};
                                break;
                            default:
                                y = {x: $.x, y: $.y}
                        }
                        var te = Q ? Le(Q) : null;
                        if (te != null) {
                            var ne = te === "y" ? "height" : "width";
                            switch (V) {
                                case De:
                                    y[te] = y[te] - ($[ne] / 2 - H[ne] / 2);
                                    break;
                                case Ve:
                                    y[te] = y[te] + ($[ne] / 2 - H[ne] / 2)
                            }
                        }
                        return y
                    }

                    var He = {top: "auto", right: "auto", bottom: "auto", left: "auto"};

                    function oe(b) {
                        var y, $ = b.popper, H = b.popperRect, z = b.placement, Q = b.variation, V = b.offsets,
                            W = b.position, K = b.gpuAcceleration, te = b.adaptive, ne = b.roundOffsets, ee = b.isFixed,
                            Ee = V.x, ae = Ee === void 0 ? 0 : Ee, he = V.y, pe = he === void 0 ? 0 : he,
                            le = typeof ne == "function" ? ne({x: ae, y: pe}) : {x: ae, y: pe};
                        ae = le.x, pe = le.y;
                        var Ce = V.hasOwnProperty("x"), ve = V.hasOwnProperty("y"), fe = re, ye = B, ce = window;
                        if (te) {
                            var ue = q($), de = "clientHeight", ge = "clientWidth";
                            ue === a($) && P(ue = _($)).position !== "static" && W === "absolute" && (de = "scrollHeight", ge = "scrollWidth"), ue = ue, (z === B || (z === re || z === ie) && Q === Ve) && (ye = N, pe -= (ee && ce.visualViewport ? ce.visualViewport.height : ue[de]) - H.height, pe *= K ? 1 : -1), z !== re && (z !== B && z !== N || Q !== Ve) || (fe = ie, ae -= (ee && ce.visualViewport ? ce.visualViewport.width : ue[ge]) - H.width, ae *= K ? 1 : -1)
                        }
                        var be, Me = Object.assign({position: W}, te && He), Re = ne === !0 ? function (Ze) {
                            var Ke = Ze.x, st = Ze.y, We = window.devicePixelRatio || 1;
                            return {x: L(Ke * We) / We || 0, y: L(st * We) / We || 0}
                        }({x: ae, y: pe}) : {x: ae, y: pe};
                        return ae = Re.x, pe = Re.y, K ? Object.assign({}, Me, ((be = {})[ye] = ve ? "0" : "", be[fe] = Ce ? "0" : "", be.transform = (ce.devicePixelRatio || 1) <= 1 ? "translate(" + ae + "px, " + pe + "px)" : "translate3d(" + ae + "px, " + pe + "px, 0)", be)) : Object.assign({}, Me, ((y = {})[ye] = ve ? pe + "px" : "", y[fe] = Ce ? ae + "px" : "", y.transform = "", y))
                    }

                    var gt = {left: "right", right: "left", bottom: "top", top: "bottom"};

                    function Rt(b) {
                        return b.replace(/left|right|bottom|top/g, function (y) {
                            return gt[y]
                        })
                    }

                    var pr = {start: "end", end: "start"};

                    function cn(b) {
                        return b.replace(/start|end/g, function (y) {
                            return pr[y]
                        })
                    }

                    function un(b, y) {
                        var $ = y.getRootNode && y.getRootNode();
                        if (b.contains(y)) return !0;
                        if ($ && k($)) {
                            var H = y;
                            do {
                                if (H && b.isSameNode(H)) return !0;
                                H = H.parentNode || H.host
                            } while (H)
                        }
                        return !1
                    }

                    function Ft(b) {
                        return Object.assign({}, b, {left: b.x, top: b.y, right: b.x + b.width, bottom: b.y + b.height})
                    }

                    function fn(b, y) {
                        return y === lt ? Ft(function ($) {
                            var H = a($), z = _($), Q = H.visualViewport, V = z.clientWidth, W = z.clientHeight, K = 0,
                                te = 0;
                            return Q && (V = Q.width, W = Q.height, /^((?!chrome|android).)*safari/i.test(navigator.userAgent) || (K = Q.offsetLeft, te = Q.offsetTop)), {
                                width: V,
                                height: W,
                                x: K + A($),
                                y: te
                            }
                        }(b)) : d(y) ? function ($) {
                            var H = m($);
                            return H.top = H.top + $.clientTop, H.left = H.left + $.clientLeft, H.bottom = H.top + $.clientHeight, H.right = H.left + $.clientWidth, H.width = $.clientWidth, H.height = $.clientHeight, H.x = H.left, H.y = H.top, H
                        }(y) : Ft(function ($) {
                            var H, z = _($), Q = h($), V = (H = $.ownerDocument) == null ? void 0 : H.body,
                                W = x(z.scrollWidth, z.clientWidth, V ? V.scrollWidth : 0, V ? V.clientWidth : 0),
                                K = x(z.scrollHeight, z.clientHeight, V ? V.scrollHeight : 0, V ? V.clientHeight : 0),
                                te = -Q.scrollLeft + A($), ne = -Q.scrollTop;
                            return P(V || z).direction === "rtl" && (te += x(z.clientWidth, V ? V.clientWidth : 0) - W), {
                                width: W,
                                height: K,
                                x: te,
                                y: ne
                            }
                        }(_(b)))
                    }

                    function dn(b) {
                        return Object.assign({}, {top: 0, right: 0, bottom: 0, left: 0}, b)
                    }

                    function pn(b, y) {
                        return y.reduce(function ($, H) {
                            return $[H] = b, $
                        }, {})
                    }

                    function Et(b, y) {
                        y === void 0 && (y = {});
                        var $ = y, H = $.placement, z = H === void 0 ? b.placement : H, Q = $.boundary,
                            V = Q === void 0 ? "clippingParents" : Q, W = $.rootBoundary, K = W === void 0 ? lt : W,
                            te = $.elementContext, ne = te === void 0 ? me : te, ee = $.altBoundary,
                            Ee = ee !== void 0 && ee, ae = $.padding, he = ae === void 0 ? 0 : ae,
                            pe = dn(typeof he != "number" ? he : pn(he, Ae)), le = ne === me ? "reference" : me,
                            Ce = b.rects.popper, ve = b.elements[Ee ? le : ne], fe = function (Re, Ze, Ke) {
                                var st = Ze === "clippingParents" ? function (ke) {
                                        var ut = S(O(ke)),
                                            Fe = ["absolute", "fixed"].indexOf(P(ke).position) >= 0 && g(ke) ? q(ke) : ke;
                                        return d(Fe) ? ut.filter(function (Ge) {
                                            return d(Ge) && un(Ge, Fe) && p(Ge) !== "body"
                                        }) : []
                                    }(Re) : [].concat(Ze), We = [].concat(st, [Ke]), Ie = We[0],
                                    qe = We.reduce(function (ke, ut) {
                                        var Fe = fn(Re, ut);
                                        return ke.top = x(Fe.top, ke.top), ke.right = D(Fe.right, ke.right), ke.bottom = D(Fe.bottom, ke.bottom), ke.left = x(Fe.left, ke.left), ke
                                    }, fn(Re, Ie));
                                return qe.width = qe.right - qe.left, qe.height = qe.bottom - qe.top, qe.x = qe.left, qe.y = qe.top, qe
                            }(d(ve) ? ve : ve.contextElement || _(b.elements.popper), V, K), ye = m(b.elements.reference),
                            ce = X({reference: ye, element: Ce, strategy: "absolute", placement: z}),
                            ue = Ft(Object.assign({}, Ce, ce)), de = ne === me ? ue : ye, ge = {
                                top: fe.top - de.top + pe.top,
                                bottom: de.bottom - fe.bottom + pe.bottom,
                                left: fe.left - de.left + pe.left,
                                right: de.right - fe.right + pe.right
                            }, be = b.modifiersData.offset;
                        if (ne === me && be) {
                            var Me = be[z];
                            Object.keys(ge).forEach(function (Re) {
                                var Ze = [ie, N].indexOf(Re) >= 0 ? 1 : -1, Ke = [B, N].indexOf(Re) >= 0 ? "y" : "x";
                                ge[Re] += Me[Ke] * Ze
                            })
                        }
                        return ge
                    }

                    function Ct(b, y, $) {
                        return x(b, D(y, $))
                    }

                    function gn(b, y, $) {
                        return $ === void 0 && ($ = {x: 0, y: 0}), {
                            top: b.top - y.height - $.y,
                            right: b.right - y.width + $.x,
                            bottom: b.bottom - y.height + $.y,
                            left: b.left - y.width - $.x
                        }
                    }

                    function hn(b) {
                        return [B, ie, N, re].some(function (y) {
                            return b[y] >= 0
                        })
                    }

                    var gr = ct({
                        defaultModifiers: [{
                            name: "eventListeners", enabled: !0, phase: "write", fn: function () {
                            }, effect: function (b) {
                                var y = b.state, $ = b.instance, H = b.options, z = H.scroll, Q = z === void 0 || z,
                                    V = H.resize, W = V === void 0 || V, K = a(y.elements.popper),
                                    te = [].concat(y.scrollParents.reference, y.scrollParents.popper);
                                return Q && te.forEach(function (ne) {
                                    ne.addEventListener("scroll", $.update, Pe)
                                }), W && K.addEventListener("resize", $.update, Pe), function () {
                                    Q && te.forEach(function (ne) {
                                        ne.removeEventListener("scroll", $.update, Pe)
                                    }), W && K.removeEventListener("resize", $.update, Pe)
                                }
                            }, data: {}
                        }, {
                            name: "popperOffsets", enabled: !0, phase: "read", fn: function (b) {
                                var y = b.state, $ = b.name;
                                y.modifiersData[$] = X({
                                    reference: y.rects.reference,
                                    element: y.rects.popper,
                                    strategy: "absolute",
                                    placement: y.placement
                                })
                            }, data: {}
                        }, {
                            name: "computeStyles", enabled: !0, phase: "beforeWrite", fn: function (b) {
                                var y = b.state, $ = b.options, H = $.gpuAcceleration, z = H === void 0 || H,
                                    Q = $.adaptive, V = Q === void 0 || Q, W = $.roundOffsets, K = W === void 0 || W,
                                    te = {
                                        placement: Be(y.placement),
                                        variation: Oe(y.placement),
                                        popper: y.elements.popper,
                                        popperRect: y.rects.popper,
                                        gpuAcceleration: z,
                                        isFixed: y.options.strategy === "fixed"
                                    };
                                y.modifiersData.popperOffsets != null && (y.styles.popper = Object.assign({}, y.styles.popper, oe(Object.assign({}, te, {
                                    offsets: y.modifiersData.popperOffsets,
                                    position: y.options.strategy,
                                    adaptive: V,
                                    roundOffsets: K
                                })))), y.modifiersData.arrow != null && (y.styles.arrow = Object.assign({}, y.styles.arrow, oe(Object.assign({}, te, {
                                    offsets: y.modifiersData.arrow,
                                    position: "absolute",
                                    adaptive: !1,
                                    roundOffsets: K
                                })))), y.attributes.popper = Object.assign({}, y.attributes.popper, {"data-popper-placement": y.placement})
                            }, data: {}
                        }, {
                            name: "applyStyles", enabled: !0, phase: "write", fn: function (b) {
                                var y = b.state;
                                Object.keys(y.elements).forEach(function ($) {
                                    var H = y.styles[$] || {}, z = y.attributes[$] || {}, Q = y.elements[$];
                                    g(Q) && p(Q) && (Object.assign(Q.style, H), Object.keys(z).forEach(function (V) {
                                        var W = z[V];
                                        W === !1 ? Q.removeAttribute(V) : Q.setAttribute(V, W === !0 ? "" : W)
                                    }))
                                })
                            }, effect: function (b) {
                                var y = b.state, $ = {
                                    popper: {position: y.options.strategy, left: "0", top: "0", margin: "0"},
                                    arrow: {position: "absolute"},
                                    reference: {}
                                };
                                return Object.assign(y.elements.popper.style, $.popper), y.styles = $, y.elements.arrow && Object.assign(y.elements.arrow.style, $.arrow), function () {
                                    Object.keys(y.elements).forEach(function (H) {
                                        var z = y.elements[H], Q = y.attributes[H] || {},
                                            V = Object.keys(y.styles.hasOwnProperty(H) ? y.styles[H] : $[H]).reduce(function (W, K) {
                                                return W[K] = "", W
                                            }, {});
                                        g(z) && p(z) && (Object.assign(z.style, V), Object.keys(Q).forEach(function (W) {
                                            z.removeAttribute(W)
                                        }))
                                    })
                                }
                            }, requires: ["computeStyles"]
                        }, {
                            name: "offset", enabled: !0, phase: "main", requires: ["popperOffsets"], fn: function (b) {
                                var y = b.state, $ = b.options, H = b.name, z = $.offset, Q = z === void 0 ? [0, 0] : z,
                                    V = ze.reduce(function (ne, ee) {
                                        return ne[ee] = function (Ee, ae, he) {
                                            var pe = Be(Ee), le = [re, B].indexOf(pe) >= 0 ? -1 : 1,
                                                Ce = typeof he == "function" ? he(Object.assign({}, ae, {placement: Ee})) : he,
                                                ve = Ce[0], fe = Ce[1];
                                            return ve = ve || 0, fe = (fe || 0) * le, [re, ie].indexOf(pe) >= 0 ? {
                                                x: fe,
                                                y: ve
                                            } : {x: ve, y: fe}
                                        }(ee, y.rects, Q), ne
                                    }, {}), W = V[y.placement], K = W.x, te = W.y;
                                y.modifiersData.popperOffsets != null && (y.modifiersData.popperOffsets.x += K, y.modifiersData.popperOffsets.y += te), y.modifiersData[H] = V
                            }
                        }, {
                            name: "flip", enabled: !0, phase: "main", fn: function (b) {
                                var y = b.state, $ = b.options, H = b.name;
                                if (!y.modifiersData[H]._skip) {
                                    for (var z = $.mainAxis, Q = z === void 0 || z, V = $.altAxis, W = V === void 0 || V, K = $.fallbackPlacements, te = $.padding, ne = $.boundary, ee = $.rootBoundary, Ee = $.altBoundary, ae = $.flipVariations, he = ae === void 0 || ae, pe = $.allowedAutoPlacements, le = y.options.placement, Ce = Be(le), ve = K || (Ce !== le && he ? function (Ge) {
                                        if (Be(Ge) === Ye) return [];
                                        var Xe = Rt(Ge);
                                        return [cn(Ge), Xe, cn(Xe)]
                                    }(le) : [Rt(le)]), fe = [le].concat(ve).reduce(function (Ge, Xe) {
                                        return Ge.concat(Be(Xe) === Ye ? function (vt, ft) {
                                            ft === void 0 && (ft = {});
                                            var et = ft, Bt = et.placement, It = et.boundary, _t = et.rootBoundary,
                                                Yt = et.padding, Jt = et.flipVariations, wt = et.allowedAutoPlacements,
                                                Kt = wt === void 0 ? ze : wt, jt = Oe(Bt),
                                                Nt = jt ? Jt ? it : it.filter(function (rt) {
                                                    return Oe(rt) === jt
                                                }) : Ae, kt = Nt.filter(function (rt) {
                                                    return Kt.indexOf(rt) >= 0
                                                });
                                            kt.length === 0 && (kt = Nt);
                                            var xt = kt.reduce(function (rt, ht) {
                                                return rt[ht] = Et(vt, {
                                                    placement: ht,
                                                    boundary: It,
                                                    rootBoundary: _t,
                                                    padding: Yt
                                                })[Be(ht)], rt
                                            }, {});
                                            return Object.keys(xt).sort(function (rt, ht) {
                                                return xt[rt] - xt[ht]
                                            })
                                        }(y, {
                                            placement: Xe,
                                            boundary: ne,
                                            rootBoundary: ee,
                                            padding: te,
                                            flipVariations: he,
                                            allowedAutoPlacements: pe
                                        }) : Xe)
                                    }, []), ye = y.rects.reference, ce = y.rects.popper, ue = new Map, de = !0, ge = fe[0], be = 0; be < fe.length; be++) {
                                        var Me = fe[be], Re = Be(Me), Ze = Oe(Me) === De, Ke = [B, N].indexOf(Re) >= 0,
                                            st = Ke ? "width" : "height", We = Et(y, {
                                                placement: Me,
                                                boundary: ne,
                                                rootBoundary: ee,
                                                altBoundary: Ee,
                                                padding: te
                                            }), Ie = Ke ? Ze ? ie : re : Ze ? N : B;
                                        ye[st] > ce[st] && (Ie = Rt(Ie));
                                        var qe = Rt(Ie), ke = [];
                                        if (Q && ke.push(We[Re] <= 0), W && ke.push(We[Ie] <= 0, We[qe] <= 0), ke.every(function (Ge) {
                                            return Ge
                                        })) {
                                            ge = Me, de = !1;
                                            break
                                        }
                                        ue.set(Me, ke)
                                    }
                                    if (de) for (var ut = function (Ge) {
                                        var Xe = fe.find(function (vt) {
                                            var ft = ue.get(vt);
                                            if (ft) return ft.slice(0, Ge).every(function (et) {
                                                return et
                                            })
                                        });
                                        if (Xe) return ge = Xe, "break"
                                    }, Fe = he ? 3 : 1; Fe > 0 && ut(Fe) !== "break"; Fe--) ;
                                    y.placement !== ge && (y.modifiersData[H]._skip = !0, y.placement = ge, y.reset = !0)
                                }
                            }, requiresIfExists: ["offset"], data: {_skip: !1}
                        }, {
                            name: "preventOverflow", enabled: !0, phase: "main", fn: function (b) {
                                var y = b.state, $ = b.options, H = b.name, z = $.mainAxis, Q = z === void 0 || z,
                                    V = $.altAxis, W = V !== void 0 && V, K = $.boundary, te = $.rootBoundary,
                                    ne = $.altBoundary, ee = $.padding, Ee = $.tether, ae = Ee === void 0 || Ee,
                                    he = $.tetherOffset, pe = he === void 0 ? 0 : he,
                                    le = Et(y, {boundary: K, rootBoundary: te, padding: ee, altBoundary: ne}),
                                    Ce = Be(y.placement), ve = Oe(y.placement), fe = !ve, ye = Le(Ce),
                                    ce = ye === "x" ? "y" : "x", ue = y.modifiersData.popperOffsets,
                                    de = y.rects.reference, ge = y.rects.popper,
                                    be = typeof pe == "function" ? pe(Object.assign({}, y.rects, {placement: y.placement})) : pe,
                                    Me = typeof be == "number" ? {
                                        mainAxis: be,
                                        altAxis: be
                                    } : Object.assign({mainAxis: 0, altAxis: 0}, be),
                                    Re = y.modifiersData.offset ? y.modifiersData.offset[y.placement] : null,
                                    Ze = {x: 0, y: 0};
                                if (ue) {
                                    if (Q) {
                                        var Ke, st = ye === "y" ? B : re, We = ye === "y" ? N : ie,
                                            Ie = ye === "y" ? "height" : "width", qe = ue[ye], ke = qe + le[st],
                                            ut = qe - le[We], Fe = ae ? -ge[Ie] / 2 : 0,
                                            Ge = ve === De ? de[Ie] : ge[Ie], Xe = ve === De ? -ge[Ie] : -de[Ie],
                                            vt = y.elements.arrow, ft = ae && vt ? f(vt) : {width: 0, height: 0},
                                            et = y.modifiersData["arrow#persistent"] ? y.modifiersData["arrow#persistent"].padding : {
                                                top: 0,
                                                right: 0,
                                                bottom: 0,
                                                left: 0
                                            }, Bt = et[st], It = et[We], _t = Ct(0, de[Ie], ft[Ie]),
                                            Yt = fe ? de[Ie] / 2 - Fe - _t - Bt - Me.mainAxis : Ge - _t - Bt - Me.mainAxis,
                                            Jt = fe ? -de[Ie] / 2 + Fe + _t + It + Me.mainAxis : Xe + _t + It + Me.mainAxis,
                                            wt = y.elements.arrow && q(y.elements.arrow),
                                            Kt = wt ? ye === "y" ? wt.clientTop || 0 : wt.clientLeft || 0 : 0,
                                            jt = (Ke = Re == null ? void 0 : Re[ye]) != null ? Ke : 0,
                                            Nt = qe + Jt - jt,
                                            kt = Ct(ae ? D(ke, qe + Yt - jt - Kt) : ke, qe, ae ? x(ut, Nt) : ut);
                                        ue[ye] = kt, Ze[ye] = kt - qe
                                    }
                                    if (W) {
                                        var xt, rt = ye === "x" ? B : re, ht = ye === "x" ? N : ie, mt = ue[ce],
                                            Qt = ce === "y" ? "height" : "width", mn = mt + le[rt], yn = mt - le[ht],
                                            Xt = [B, re].indexOf(Ce) !== -1,
                                            bn = (xt = Re == null ? void 0 : Re[ce]) != null ? xt : 0,
                                            vn = Xt ? mn : mt - de[Qt] - ge[Qt] - bn + Me.altAxis,
                                            _n = Xt ? mt + de[Qt] + ge[Qt] - bn - Me.altAxis : yn,
                                            wn = ae && Xt ? function (hr, mr, en) {
                                                var kn = Ct(hr, mr, en);
                                                return kn > en ? en : kn
                                            }(vn, mt, _n) : Ct(ae ? vn : mn, mt, ae ? _n : yn);
                                        ue[ce] = wn, Ze[ce] = wn - mt
                                    }
                                    y.modifiersData[H] = Ze
                                }
                            }, requiresIfExists: ["offset"]
                        }, {
                            name: "arrow", enabled: !0, phase: "main", fn: function (b) {
                                var y, $ = b.state, H = b.name, z = b.options, Q = $.elements.arrow,
                                    V = $.modifiersData.popperOffsets, W = Be($.placement), K = Le(W),
                                    te = [re, ie].indexOf(W) >= 0 ? "height" : "width";
                                if (Q && V) {
                                    var ne = function (ge, be) {
                                            return dn(typeof (ge = typeof ge == "function" ? ge(Object.assign({}, be.rects, {placement: be.placement})) : ge) != "number" ? ge : pn(ge, Ae))
                                        }(z.padding, $), ee = f(Q), Ee = K === "y" ? B : re, ae = K === "y" ? N : ie,
                                        he = $.rects.reference[te] + $.rects.reference[K] - V[K] - $.rects.popper[te],
                                        pe = V[K] - $.rects.reference[K], le = q(Q),
                                        Ce = le ? K === "y" ? le.clientHeight || 0 : le.clientWidth || 0 : 0,
                                        ve = he / 2 - pe / 2, fe = ne[Ee], ye = Ce - ee[te] - ne[ae],
                                        ce = Ce / 2 - ee[te] / 2 + ve, ue = Ct(fe, ce, ye), de = K;
                                    $.modifiersData[H] = ((y = {})[de] = ue, y.centerOffset = ue - ce, y)
                                }
                            }, effect: function (b) {
                                var y = b.state, $ = b.options.element, H = $ === void 0 ? "[data-popper-arrow]" : $;
                                H != null && (typeof H != "string" || (H = y.elements.popper.querySelector(H))) && un(y.elements.popper, H) && (y.elements.arrow = H)
                            }, requires: ["popperOffsets"], requiresIfExists: ["preventOverflow"]
                        }, {
                            name: "hide",
                            enabled: !0,
                            phase: "main",
                            requiresIfExists: ["preventOverflow"],
                            fn: function (b) {
                                var y = b.state, $ = b.name, H = y.rects.reference, z = y.rects.popper,
                                    Q = y.modifiersData.preventOverflow, V = Et(y, {elementContext: "reference"}),
                                    W = Et(y, {altBoundary: !0}), K = gn(V, H), te = gn(W, z, Q), ne = hn(K),
                                    ee = hn(te);
                                y.modifiersData[$] = {
                                    referenceClippingOffsets: K,
                                    popperEscapeOffsets: te,
                                    isReferenceHidden: ne,
                                    hasPopperEscaped: ee
                                }, y.attributes.popper = Object.assign({}, y.attributes.popper, {
                                    "data-popper-reference-hidden": ne,
                                    "data-popper-escaped": ee
                                })
                            }
                        }]
                    })
                }
            }, t = {};

            function o(s) {
                var l = t[s];
                if (l !== void 0) return l.exports;
                var c = t[s] = {exports: {}};
                return n[s](c, c.exports, o), c.exports
            }

            o.d = (s, l) => {
                for (var c in l) o.o(l, c) && !o.o(s, c) && Object.defineProperty(s, c, {enumerable: !0, get: l[c]})
            }, o.o = (s, l) => Object.prototype.hasOwnProperty.call(s, l), o.r = s => {
                typeof Symbol < "u" && Symbol.toStringTag && Object.defineProperty(s, Symbol.toStringTag, {value: "Module"}), Object.defineProperty(s, "__esModule", {value: !0})
            };
            var i = {};
            return o.r(i), o(661), o(795), o(682), o(284), o(181), o(778), o(51), o(185), i
        })()
    })
})(Hr);

function Mr(r) {
    let e;
    return {
        c() {
            e = E("div"), e.innerHTML = '<h1 class="text-3xl font-bold text-gray-800 sm:text-4xl dark:text-white">欢迎使用Vanna.AI</h1> <p class="mt-3 text-gray-600 dark:text-gray-400"></p>', v(e, "class", "max-w-4xl px-4 sm:px-6 lg:px-8 mx-auto text-center")
        }, m(n, t) {
            U(n, e, t)
        }, p: se, i: se, o: se, d(n) {
            n && G(e)
        }
    }
}

class Rr extends Se {
    constructor(e) {
        super(), $e(this, e, null, Mr, xe, {})
    }
}

function Br(r) {
    let e, n;
    const t = r[1].default, o = Gt(t, r, r[0], null);
    return {
        c() {
            e = E("p"), o && o.c(), v(e, "class", "text-gray-800 dark:text-gray-200")
        }, m(i, s) {
            U(i, e, s), o && o.m(e, null), n = !0
        }, p(i, [s]) {
            o && o.p && (!n || s & 1) && Zt(o, t, i, i[0], n ? Ut(t, i[0], s, null) : Wt(i[0]), null)
        }, i(i) {
            n || (M(o, i), n = !0)
        }, o(i) {
            I(o, i), n = !1
        }, d(i) {
            i && G(e), o && o.d(i)
        }
    }
}

function Ir(r, e, n) {
    let {$$slots: t = {}, $$scope: o} = e;
    return r.$$set = i => {
        "$$scope" in i && n(0, o = i.$$scope)
    }, [o, t]
}

class pt extends Se {
    constructor(e) {
        super(), $e(this, e, Ir, Br, xe, {})
    }
}

function Nr(r) {
    let e;
    return {
        c() {
            e = we(r[0])
        }, m(n, t) {
            U(n, e, t)
        }, p(n, t) {
            t & 1 && Ue(e, n[0])
        }, d(n) {
            n && G(e)
        }
    }
}

function Qr(r) {
    let e, n, t, o, i, s, l, c, a;
    l = new pt({props: {$$slots: {default: [Nr]}, $$scope: {ctx: r}}});
    const d = r[1].default, g = Gt(d, r, r[2], null);
    return {
        c() {
            e = E("li"), n = E("div"), t = E("div"), o = E("span"), o.innerHTML = '<span class="text-sm font-medium text-white leading-none">你</span>', i = Z(), s = E("div"), J(l.$$.fragment), c = Z(), g && g.c(), v(o, "class", "flex-shrink-0 inline-flex items-center justify-center h-[2.375rem] w-[2.375rem] rounded-full bg-gray-600"), v(s, "class", "grow mt-2 space-y-3"), v(t, "class", "max-w-2xl flex gap-x-2 sm:gap-x-4"), v(n, "class", "max-w-4xl px-4 sm:px-6 lg:px-8 mx-auto"), v(e, "class", "py-2 sm:py-4")
        }, m(k, x) {
            U(k, e, x), w(e, n), w(n, t), w(t, o), w(t, i), w(t, s), F(l, s, null), w(s, c), g && g.m(s, null), a = !0
        }, p(k, [x]) {
            const D = {};
            x & 5 && (D.$$scope = {
                dirty: x,
                ctx: k
            }), l.$set(D), g && g.p && (!a || x & 4) && Zt(g, d, k, k[2], a ? Ut(d, k[2], x, null) : Wt(k[2]), null)
        }, i(k) {
            a || (M(l.$$.fragment, k), M(g, k), a = !0)
        }, o(k) {
            I(l.$$.fragment, k), I(g, k), a = !1
        }, d(k) {
            k && G(e), Y(l), g && g.d(k)
        }
    }
}

function Vr(r, e, n) {
    let {$$slots: t = {}, $$scope: o} = e, {message: i} = e;
    return r.$$set = s => {
        "message" in s && n(0, i = s.message), "$$scope" in s && n(2, o = s.$$scope)
    }, [i, t, o]
}

class Mt extends Se {
    constructor(e) {
        super(), $e(this, e, Vr, Qr, xe, {message: 0})
    }
}

function zr(r) {
    let e, n, t, o, i, s, l, c, a, d, g;
    return {
        c() {
            e = E("div"), n = E("input"), t = Z(), o = E("div"), i = E("div"), s = E("div"), s.innerHTML = "", l = Z(), c = E("div"), a = E("button"), a.innerHTML = '<svg class="h-3.5 w-3.5" xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M15.964.686a.5.5 0 0 0-.65-.65L.767 5.855H.766l-.452.18a.5.5 0 0 0-.082.887l.41.26.001.002 4.995 3.178 3.178 4.995.002.002.26.41a.5.5 0 0 0 .886-.083l6-15Zm-1.833 1.89L6.637 10.07l-.215-.338a.5.5 0 0 0-.154-.154l-.338-.215 7.494-7.494 1.178-.471-.47 1.178Z"></path></svg>', v(n, "type", "text"), v(n, "class", "p-4 pb-12 block w-full bg-gray-100 border-gray-200 rounded-md text-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-slate-800 dark:border-gray-700 dark:text-gray-400"), v(n, "placeholder", "向我询问一个有关您可以将其转化为SQL的数据的问题."), v(s, "class", "flex items-center"), v(a, "type", "button"), v(a, "class", "inline-flex flex-shrink-0 justify-center items-center h-8 w-8 rounded-md text-white bg-blue-600 hover:bg-blue-500 focus:z-10 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"), v(c, "class", "flex items-center gap-x-1"), v(i, "class", "flex justify-between items-center"), v(o, "class", "absolute bottom-px inset-x-px p-2 rounded-b-md bg-gray-100 dark:bg-slate-800"), v(e, "class", "relative")
        }, m(k, x) {
            U(k, e, x), w(e, n), bt(n, r[0]), w(e, t), w(e, o), w(o, i), w(i, s), w(i, l), w(i, c), w(c, a), d || (g = [je(n, "input", r[4]), je(n, "keydown", r[1]), je(a, "click", r[2])], d = !0)
        }, p(k, [x]) {
            x & 1 && n.value !== k[0] && bt(n, k[0])
        }, i: se, o: se, d(k) {
            k && G(e), d = !1, ot(g)
        }
    }
}

function Gr(r, e, n) {
    let {onSubmit: t} = e, o = "";

    function i(c) {
        c.key === "Enter" && (t(o), c.preventDefault())
    }

    function s() {
        t(o)
    }

    function l() {
        o = this.value, n(0, o)
    }

    return r.$$set = c => {
        "onSubmit" in c && n(3, t = c.onSubmit)
    }, [o, i, s, t, l]
}

class Ur extends Se {
    constructor(e) {
        super(), $e(this, e, Gr, zr, xe, {onSubmit: 3})
    }
}

function Zr(r) {
    let e;
    return {
        c() {
            e = E("div"), e.innerHTML = '<button type="button" class="p-2 inline-flex justify-center items-center gap-1.5 rounded-md border font-medium bg-white text-gray-700 shadow-sm align-middle hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-white focus:ring-blue-600 transition-all text-xs dark:bg-slate-900 dark:hover:bg-slate-800 dark:border-gray-700 dark:text-gray-400 dark:hover:text-white dark:focus:ring-offset-gray-800" data-hs-overlay="#application-sidebar" aria-controls="application-sidebar" aria-label="Toggle navigation"><svg class="w-3.5 h-3.5" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M2.5 12a.5.5 0 0 1 .5-.5h10a.5.5 0 0 1 0 1H3a.5.5 0 0 1-.5-.5zm0-4a.5.5 0 0 1 .5-.5h10a.5.5 0 0 1 0 1H3a.5.5 0 0 1-.5-.5zm0-4a.5.5 0 0 1 .5-.5h10a.5.5 0 0 1 0 1H3a.5.5 0 0 1-.5-.5z"></path></svg> <span>Sidebar</span></button>', v(e, "class", "lg:hidden flex justify-end mb-2 sm:mb-3")
        }, m(n, t) {
            U(n, e, t)
        }, p: se, i: se, o: se, d(n) {
            n && G(e)
        }
    }
}

class Wr extends Se {
    constructor(e) {
        super(), $e(this, e, null, Zr, xe, {})
    }
}

function Fr(r) {
    let e, n, t, o;
    return {
        c() {
            e = E("button"), n = we(r[0]), v(e, "type", "button"), v(e, "class", "mb-2.5 mr-1.5 py-2 px-3 inline-flex justify-center items-center gap-x-2 rounded-md border border-blue-600 bg-white text-blue-600 align-middle hover:bg-blue-50 text-sm dark:bg-slate-900 dark:text-blue-500 dark:border-blue-500 dark:hover:text-blue-400 dark:hover:border-blue-400")
        }, m(i, s) {
            U(i, e, s), w(e, n), t || (o = je(e, "click", r[1]), t = !0)
        }, p(i, [s]) {
            s & 1 && Ue(n, i[0])
        }, i: se, o: se, d(i) {
            i && G(e), t = !1, o()
        }
    }
}

function Yr(r, e, n) {
    let {message: t} = e, {onSubmit: o} = e;

    function i() {
        o(t)
    }

    return r.$$set = s => {
        "message" in s && n(0, t = s.message), "onSubmit" in s && n(2, o = s.onSubmit)
    }, [t, i, o]
}

class Ht extends Se {
    constructor(e) {
        super(), $e(this, e, Yr, Fr, xe, {message: 0, onSubmit: 2})
    }
}

function Jr(r) {
    let e, n, t, o, i, s;
    const l = r[1].default, c = Gt(l, r, r[0], null);
    return {
        c() {
            e = E("li"), n = E("img"), o = Z(), i = E("div"), c && c.c(), or(n.src, t = "/vanna.svg") || v(n, "src", t), v(n, "class", "flex-shrink-0 w-[2.375rem] h-[2.375rem] "), v(n, "alt", "agent logo"), v(i, "class", "space-y-3 overflow-x-auto overflow-y-hidden"), v(e, "class", "max-w-4xl py-2 px-4 sm:px-6 lg:px-8 mx-auto flex gap-x-2 sm:gap-x-4")
        }, m(a, d) {
            U(a, e, d), w(e, n), w(e, o), w(e, i), c && c.m(i, null), s = !0
        }, p(a, [d]) {
            c && c.p && (!s || d & 1) && Zt(c, l, a, a[0], s ? Ut(l, a[0], d, null) : Wt(a[0]), null)
        }, i(a) {
            s || (M(c, a), s = !0)
        }, o(a) {
            I(c, a), s = !1
        }, d(a) {
            a && G(e), c && c.d(a)
        }
    }
}

function Kr(r, e, n) {
    let {$$slots: t = {}, $$scope: o} = e;
    return r.$$set = i => {
        "$$scope" in i && n(0, o = i.$$scope)
    }, [o, t]
}

class tt extends Se {
    constructor(e) {
        super(), $e(this, e, Kr, Jr, xe, {})
    }
}

function Xr(r) {
    let e;
    return {
        c() {
            e = we("思考中...")
        }, m(n, t) {
            U(n, e, t)
        }, d(n) {
            n && G(e)
        }
    }
}

function eo(r) {
    let e, n, t, o, i, s, l;
    return s = new pt({props: {$$slots: {default: [Xr]}, $$scope: {ctx: r}}}), {
        c() {
            e = E("li"), n = E("img"), o = Z(), i = E("div"), J(s.$$.fragment), or(n.src, t = "/vanna.svg") || v(n, "src", t), v(n, "class", "flex-shrink-0 w-[2.375rem] h-[2.375rem] animate-bounce "), v(n, "alt", "agent logo"), v(i, "class", "space-y-3"), v(e, "class", "max-w-4xl py-2 px-4 sm:px-6 lg:px-8 mx-auto flex gap-x-2 sm:gap-x-4")
        }, m(c, a) {
            U(c, e, a), w(e, n), w(e, o), w(e, i), F(s, i, null), l = !0
        }, p(c, [a]) {
            const d = {};
            a & 1 && (d.$$scope = {dirty: a, ctx: c}), s.$set(d)
        }, i(c) {
            l || (M(s.$$.fragment, c), l = !0)
        }, o(c) {
            I(s.$$.fragment, c), l = !1
        }, d(c) {
            c && G(e), Y(s)
        }
    }
}

class to extends Se {
    constructor(e) {
        super(), $e(this, e, null, eo, xe, {})
    }
}

function no(r) {
    let e, n, t, o, i, s, l, c, a, d, g;
    return {
        c() {
            e = E("ul"), n = E("li"), t = E("div"), o = E("span"), o.textContent = "CSV", i = Z(), s = E("a"), l = Pt("svg"), c = Pt("path"), a = Pt("path"), d = we(`
          Download`), v(o, "class", "mr-3 flex-1 w-0 truncate"), v(c, "d", "M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"), v(a, "d", "M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"), v(l, "class", "flex-shrink-0 w-3 h-3"), v(l, "width", "16"), v(l, "height", "16"), v(l, "viewBox", "0 0 16 16"), v(l, "fill", "currentColor"), v(s, "class", "flex items-center gap-x-2 text-gray-500 hover:text-blue-500 whitespace-nowrap"), v(s, "href", g = "/api/v0/download_csv?id=" + r[0]), v(t, "class", "w-full flex justify-between truncate"), v(n, "class", "flex items-center gap-x-2 p-3 text-sm bg-white border text-gray-800 first:rounded-t-lg first:mt-0 last:rounded-b-lg dark:bg-slate-900 dark:border-gray-700 dark:text-gray-200"), v(e, "class", "flex flex-col justify-end text-start -space-y-px")
        }, m(k, x) {
            U(k, e, x), w(e, n), w(n, t), w(t, o), w(t, i), w(t, s), w(s, l), w(l, c), w(l, a), w(s, d)
        }, p(k, [x]) {
            x & 1 && g !== (g = "/api/v0/download_csv?id=" + k[0]) && v(s, "href", g)
        }, i: se, o: se, d(k) {
            k && G(e)
        }
    }
}

function ro(r, e, n) {
    let {id: t} = e;
    return r.$$set = o => {
        "id" in o && n(0, t = o.id)
    }, [t]
}

class oo extends Se {
    constructor(e) {
        super(), $e(this, e, ro, no, xe, {id: 0})
    }
}

function En(r, e, n) {
    const t = r.slice();
    return t[4] = e[n], t
}

function Cn(r, e, n) {
    const t = r.slice();
    return t[7] = e[n], t
}

function jn(r, e, n) {
    const t = r.slice();
    return t[7] = e[n], t
}

function Pn(r) {
    let e, n, t, o;
    return {
        c() {
            e = E("th"), n = E("div"), t = E("span"), t.textContent = `${r[7]}`, o = Z(), v(t, "class", "text-xs font-semibold uppercase tracking-wide text-gray-800 dark:text-gray-200"), v(n, "class", "flex items-center gap-x-2"), v(e, "scope", "col"), v(e, "class", "px-6 py-3 text-left")
        }, m(i, s) {
            U(i, e, s), w(e, n), w(n, t), w(e, o)
        }, p: se, d(i) {
            i && G(e)
        }
    }
}

function qn(r) {
    let e, n, t;
    return {
        c() {
            e = E("td"), n = E("div"), t = E("span"), t.textContent = `${r[4][r[7]]}`, v(t, "class", "text-gray-800 dark:text-gray-200"), v(n, "class", "px-6 py-3"), v(e, "class", "h-px w-px whitespace-nowrap")
        }, m(o, i) {
            U(o, e, i), w(e, n), w(n, t)
        }, p: se, d(o) {
            o && G(e)
        }
    }
}

function An(r) {
    let e, n, t = _e(r[2]), o = [];
    for (let i = 0; i < t.length; i += 1) o[i] = qn(Cn(r, t, i));
    return {
        c() {
            e = E("tr");
            for (let i = 0; i < o.length; i += 1) o[i].c();
            n = Z()
        }, m(i, s) {
            U(i, e, s);
            for (let l = 0; l < o.length; l += 1) o[l] && o[l].m(e, null);
            w(e, n)
        }, p(i, s) {
            if (s & 6) {
                t = _e(i[2]);
                let l;
                for (l = 0; l < t.length; l += 1) {
                    const c = Cn(i, t, l);
                    o[l] ? o[l].p(c, s) : (o[l] = qn(c), o[l].c(), o[l].m(e, n))
                }
                for (; l < o.length; l += 1) o[l].d(1);
                o.length = t.length
            }
        }, d(i) {
            i && G(e), Je(o, i)
        }
    }
}

function io(r) {
    let e, n, t, o, i, s, l, c, a, d, g, k = _e(r[2]), x = [];
    for (let m = 0; m < k.length; m += 1) x[m] = Pn(jn(r, k, m));
    let D = _e(r[1]), L = [];
    for (let m = 0; m < D.length; m += 1) L[m] = An(En(r, D, m));
    return d = new oo({props: {id: r[0]}}), {
        c() {
            e = E("div"), n = E("div"), t = E("div"), o = E("table"), i = E("thead"), s = E("tr");
            for (let m = 0; m < x.length; m += 1) x[m].c();
            l = Z(), c = E("tbody");
            for (let m = 0; m < L.length; m += 1) L[m].c();
            a = Z(), J(d.$$.fragment), v(i, "class", "bg-gray-50 dark:bg-slate-800"), v(c, "class", "divide-y divide-gray-200 dark:divide-gray-700"), v(o, "class", "min-w-full divide-y divide-gray-200 dark:divide-gray-700"), v(t, "class", "p-1.5 min-w-full inline-block align-middle"), v(n, "class", "-m-1.5 overflow-x-auto"), v(e, "class", "bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden dark:bg-slate-900 dark:border-gray-700")
        }, m(m, h) {
            U(m, e, h), w(e, n), w(n, t), w(t, o), w(o, i), w(i, s);
            for (let p = 0; p < x.length; p += 1) x[p] && x[p].m(s, null);
            w(o, l), w(o, c);
            for (let p = 0; p < L.length; p += 1) L[p] && L[p].m(c, null);
            U(m, a, h), F(d, m, h), g = !0
        }, p(m, [h]) {
            if (h & 4) {
                k = _e(m[2]);
                let _;
                for (_ = 0; _ < k.length; _ += 1) {
                    const A = jn(m, k, _);
                    x[_] ? x[_].p(A, h) : (x[_] = Pn(A), x[_].c(), x[_].m(s, null))
                }
                for (; _ < x.length; _ += 1) x[_].d(1);
                x.length = k.length
            }
            if (h & 6) {
                D = _e(m[1]);
                let _;
                for (_ = 0; _ < D.length; _ += 1) {
                    const A = En(m, D, _);
                    L[_] ? L[_].p(A, h) : (L[_] = An(A), L[_].c(), L[_].m(c, null))
                }
                for (; _ < L.length; _ += 1) L[_].d(1);
                L.length = D.length
            }
            const p = {};
            h & 1 && (p.id = m[0]), d.$set(p)
        }, i(m) {
            g || (M(d.$$.fragment, m), g = !0)
        }, o(m) {
            I(d.$$.fragment, m), g = !1
        }, d(m) {
            m && (G(e), G(a)), Je(x, m), Je(L, m), Y(d, m)
        }
    }
}

function so(r, e, n) {
    let {id: t} = e, {df: o} = e, i = JSON.parse(o), s = i.length > 0 ? Object.keys(i[0]) : [];
    return r.$$set = l => {
        "id" in l && n(0, t = l.id), "df" in l && n(3, o = l.df)
    }, [t, i, s, o]
}

class ar extends Se {
    constructor(e) {
        super(), $e(this, e, so, io, xe, {id: 0, df: 3})
    }
}

function lo(r) {
    let e;
    return {
        c() {
            e = E("div"), v(e, "id", r[0])
        }, m(n, t) {
            U(n, e, t)
        }, p: se, i: se, o: se, d(n) {
            n && G(e)
        }
    }
}

function ao(r, e, n) {
    let {fig: t} = e, o = JSON.parse(t),
        i = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
    return sr(() => {
        Plotly.newPlot(document.getElementById(i), o, {responsive: !0})
    }), r.$$set = s => {
        "fig" in s && n(1, t = s.fig)
    }, [i, t]
}

class cr extends Se {
    constructor(e) {
        super(), $e(this, e, ao, lo, xe, {fig: 1})
    }
}

function co(r) {
    let e, n, t, o;
    return {
        c() {
            e = E("button"), n = we(r[0]), v(e, "type", "button"), v(e, "class", "mb-2.5 mr-1.5 py-3 px-4 inline-flex justify-center items-center gap-2 rounded-md border-2 border-green-200 font-semibold text-green-500 hover:text-white hover:bg-green-500 hover:border-green-500 focus:outline-none focus:ring-2 focus:ring-green-200 focus:ring-offset-2 transition-all text-sm dark:focus:ring-offset-gray-800")
        }, m(i, s) {
            U(i, e, s), w(e, n), t || (o = je(e, "click", r[1]), t = !0)
        }, p(i, [s]) {
            s & 1 && Ue(n, i[0])
        }, i: se, o: se, d(i) {
            i && G(e), t = !1, o()
        }
    }
}

function uo(r, e, n) {
    let {message: t} = e, {onSubmit: o} = e;

    function i() {
        o(t)
    }

    return r.$$set = s => {
        "message" in s && n(0, t = s.message), "onSubmit" in s && n(2, o = s.onSubmit)
    }, [t, i, o]
}

class ur extends Se {
    constructor(e) {
        super(), $e(this, e, uo, co, xe, {message: 0, onSubmit: 2})
    }
}

function fo(r) {
    let e, n, t, o, i, s, l, c, a;
    return {
        c() {
            e = E("div"), n = E("div"), t = E("div"), t.innerHTML = '<svg class="h-4 w-4 text-yellow-400 mt-0.5" xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2z"></path></svg>', o = Z(), i = E("div"), s = E("h3"), s.textContent = "出错", l = Z(), c = E("div"), a = we(r[0]), v(t, "class", "flex-shrink-0"), v(s, "class", "text-sm text-yellow-800 font-semibold"), v(c, "class", "mt-1 text-sm text-yellow-700"), v(i, "class", "ml-4"), v(n, "class", "flex"), v(e, "class", "bg-yellow-50 border border-yellow-200 rounded-md p-4"), v(e, "role", "alert")
        }, m(d, g) {
            U(d, e, g), w(e, n), w(n, t), w(n, o), w(n, i), w(i, s), w(i, l), w(i, c), w(c, a)
        }, p(d, [g]) {
            g & 1 && Ue(a, d[0])
        }, i: se, o: se, d(d) {
            d && G(e)
        }
    }
}

function po(r, e, n) {
    let {message: t} = e;
    return r.$$set = o => {
        "message" in o && n(0, t = o.message)
    }, [t]
}

let fr = class extends Se {
    constructor(e) {
        super(), $e(this, e, po, fo, xe, {message: 0})
    }
};

function go(r) {
    let e, n;
    const t = r[1].default, o = Gt(t, r, r[0], null);
    return {
        c() {
            e = E("div"), o && o.c(), v(e, "class", "font-mono whitespace-pre-wrap")
        }, m(i, s) {
            U(i, e, s), o && o.m(e, null), n = !0
        }, p(i, [s]) {
            o && o.p && (!n || s & 1) && Zt(o, t, i, i[0], n ? Ut(t, i[0], s, null) : Wt(i[0]), null)
        }, i(i) {
            n || (M(o, i), n = !0)
        }, o(i) {
            I(o, i), n = !1
        }, d(i) {
            i && G(e), o && o.d(i)
        }
    }
}

function ho(r, e, n) {
    let {$$slots: t = {}, $$scope: o} = e;
    return r.$$set = i => {
        "$$scope" in i && n(0, o = i.$$scope)
    }, [o, t]
}

class dr extends Se {
    constructor(e) {
        super(), $e(this, e, ho, go, xe, {})
    }
}

function mo(r) {
    let e, n, t, o, i, s;
    return t = new Ht({props: {message: "训练", onSubmit: r[3]}}), {
        c() {
            e = E("textarea"), n = Z(), J(t.$$.fragment), v(e, "class", "block p-2.5 w-full text-blue-600 hover:text-blue-500 text-2xl dark:text-blue-500 dark:hover:text-blue-400 bg-gray-50 rounded-lg border border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500 font-mono"), v(e, "placeholder", "SELECT col1, col2, col3 FROM ...")
        }, m(l, c) {
            U(l, e, c), bt(e, r[1]), U(l, n, c), F(t, l, c), o = !0, i || (s = je(e, "input", r[2]), i = !0)
        }, p(l, [c]) {
            c & 2 && bt(e, l[1]);
            const a = {};
            c & 3 && (a.onSubmit = l[3]), t.$set(a)
        }, i(l) {
            o || (M(t.$$.fragment, l), o = !0)
        }, o(l) {
            I(t.$$.fragment, l), o = !1
        }, d(l) {
            l && (G(e), G(n)), Y(t, l), i = !1, s()
        }
    }
}

function yo(r, e, n) {
    let {onSubmit: t} = e, o;

    function i() {
        o = this.value, n(1, o)
    }

    const s = () => t(o);
    return r.$$set = l => {
        "onSubmit" in l && n(0, t = l.onSubmit)
    }, [t, o, i, s]
}

class bo extends Se {
    constructor(e) {
        super(), $e(this, e, yo, mo, xe, {onSubmit: 0})
    }
}

function Dn(r, e, n) {
    const t = r.slice();
    return t[12] = e[n], t
}

function Hn(r, e, n) {
    const t = r.slice();
    return t[15] = e[n], t
}

function Mn(r, e, n) {
    const t = r.slice();
    return t[18] = e[n], t
}

function Rn(r, e, n) {
    const t = r.slice();
    return t[18] = e[n], t
}

function Bn(r) {
    let e, n;
    return e = new tt({props: {$$slots: {default: [_o]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8388618 && (i.$$scope = {dirty: o, ctx: t}), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function In(r) {
    let e, n;
    return e = new Ht({props: {message: r[18], onSubmit: r[3]}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 2 && (i.message = t[18]), o & 8 && (i.onSubmit = t[3]), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function vo(r) {
    let e = r[1].header + "", n, t, o, i, s = _e(r[1].questions), l = [];
    for (let a = 0; a < s.length; a += 1) l[a] = In(Rn(r, s, a));
    const c = a => I(l[a], 1, 1, () => {
        l[a] = null
    });
    return {
        c() {
            n = we(e), t = Z();
            for (let a = 0; a < l.length; a += 1) l[a].c();
            o = nt()
        }, m(a, d) {
            U(a, n, d), U(a, t, d);
            for (let g = 0; g < l.length; g += 1) l[g] && l[g].m(a, d);
            U(a, o, d), i = !0
        }, p(a, d) {
            if ((!i || d & 2) && e !== (e = a[1].header + "") && Ue(n, e), d & 10) {
                s = _e(a[1].questions);
                let g;
                for (g = 0; g < s.length; g += 1) {
                    const k = Rn(a, s, g);
                    l[g] ? (l[g].p(k, d), M(l[g], 1)) : (l[g] = In(k), l[g].c(), M(l[g], 1), l[g].m(o.parentNode, o))
                }
                for (Ne(), g = s.length; g < l.length; g += 1) c(g);
                Qe()
            }
        }, i(a) {
            if (!i) {
                for (let d = 0; d < s.length; d += 1) M(l[d]);
                i = !0
            }
        }, o(a) {
            l = l.filter(Boolean);
            for (let d = 0; d < l.length; d += 1) I(l[d]);
            i = !1
        }, d(a) {
            a && (G(n), G(t), G(o)), Je(l, a)
        }
    }
}

function _o(r) {
    let e, n;
    return e = new pt({props: {$$slots: {default: [vo]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8388618 && (i.$$scope = {dirty: o, ctx: t}), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function wo(r) {
    let e, n;
    return e = new tt({props: {$$slots: {default: [jo]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8388612 && (i.$$scope = {dirty: o, ctx: t}), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function ko(r) {
    let e, n;
    return e = new Mt({props: {message: "将您的SQL放在这里", $$slots: {default: [Po]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8388672 && (i.$$scope = {dirty: o, ctx: t}), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function xo(r) {
    let e, n, t, o, i, s, l, c;
    return e = new Mt({props: {message: r[15].question}}), t = new tt({
        props: {
            $$slots: {default: [Do]},
            $$scope: {ctx: r}
        }
    }), i = new tt({
        props: {
            $$slots: {default: [Ho]},
            $$scope: {ctx: r}
        }
    }), l = new tt({props: {$$slots: {default: [Mo]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment), n = Z(), J(t.$$.fragment), o = Z(), J(i.$$.fragment), s = Z(), J(l.$$.fragment)
        }, m(a, d) {
            F(e, a, d), U(a, n, d), F(t, a, d), U(a, o, d), F(i, a, d), U(a, s, d), F(l, a, d), c = !0
        }, p(a, d) {
            const g = {};
            d & 4 && (g.message = a[15].question), e.$set(g);
            const k = {};
            d & 8388612 && (k.$$scope = {dirty: d, ctx: a}), t.$set(k);
            const x = {};
            d & 8388612 && (x.$$scope = {dirty: d, ctx: a}), i.$set(x);
            const D = {};
            d & 8388612 && (D.$$scope = {dirty: d, ctx: a}), l.$set(D)
        }, i(a) {
            c || (M(e.$$.fragment, a), M(t.$$.fragment, a), M(i.$$.fragment, a), M(l.$$.fragment, a), c = !0)
        }, o(a) {
            I(e.$$.fragment, a), I(t.$$.fragment, a), I(i.$$.fragment, a), I(l.$$.fragment, a), c = !1
        }, d(a) {
            a && (G(n), G(o), G(s)), Y(e, a), Y(t, a), Y(i, a), Y(l, a)
        }
    }
}

function $o(r) {
    let e, n;
    return e = new tt({props: {$$slots: {default: [Ro]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8388612 && (i.$$scope = {dirty: o, ctx: t}), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function So(r) {
    let e, n, t, o, i, s, l, c;
    e = new tt({props: {$$slots: {default: [Bo]}, $$scope: {ctx: r}}}), t = new tt({
        props: {
            $$slots: {default: [No]},
            $$scope: {ctx: r}
        }
    });
    const a = [Vo, Qo], d = [];

    function g(k, x) {
        return k[0] === !0 ? 0 : k[0] === !1 ? 1 : -1
    }

    return ~(i = g(r)) && (s = d[i] = a[i](r)), {
        c() {
            J(e.$$.fragment), n = Z(), J(t.$$.fragment), o = Z(), s && s.c(), l = nt()
        }, m(k, x) {
            F(e, k, x), U(k, n, x), F(t, k, x), U(k, o, x), ~i && d[i].m(k, x), U(k, l, x), c = !0
        }, p(k, x) {
            const D = {};
            x & 8388612 && (D.$$scope = {dirty: x, ctx: k}), e.$set(D);
            const L = {};
            x & 8388609 && (L.$$scope = {dirty: x, ctx: k}), t.$set(L);
            let m = i;
            i = g(k), i !== m && (s && (Ne(), I(d[m], 1, 1, () => {
                d[m] = null
            }), Qe()), ~i ? (s = d[i], s || (s = d[i] = a[i](k), s.c()), M(s, 1), s.m(l.parentNode, l)) : s = null)
        }, i(k) {
            c || (M(e.$$.fragment, k), M(t.$$.fragment, k), M(s), c = !0)
        }, o(k) {
            I(e.$$.fragment, k), I(t.$$.fragment, k), I(s), c = !1
        }, d(k) {
            k && (G(n), G(o), G(l)), Y(e, k), Y(t, k), ~i && d[i].d(k)
        }
    }
}

function Oo(r) {
    let e, n;
    return e = new tt({props: {$$slots: {default: [zo]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8388612 && (i.$$scope = {dirty: o, ctx: t}), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Lo(r) {
    let e, n;
    return e = new tt({props: {$$slots: {default: [Uo]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8388620 && (i.$$scope = {dirty: o, ctx: t}), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function To(r) {
    let e, n;
    return e = new tt({props: {$$slots: {default: [Fo]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8388612 && (i.$$scope = {dirty: o, ctx: t}), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Eo(r) {
    let e, n;
    return e = new Mt({props: {message: r[15].question}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 4 && (i.message = t[15].question), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Co(r) {
    let e = JSON.stringify(r[15]) + "", n;
    return {
        c() {
            n = we(e)
        }, m(t, o) {
            U(t, n, o)
        }, p(t, o) {
            o & 4 && e !== (e = JSON.stringify(t[15]) + "") && Ue(n, e)
        }, d(t) {
            t && G(n)
        }
    }
}

function jo(r) {
    let e, n;
    return e = new pt({props: {$$slots: {default: [Co]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8388612 && (i.$$scope = {dirty: o, ctx: t}), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Po(r) {
    let e, n;
    return e = new bo({props: {onSubmit: r[6]}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 64 && (i.onSubmit = t[6]), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function qo(r) {
    let e = r[15].sql + "", n;
    return {
        c() {
            n = we(e)
        }, m(t, o) {
            U(t, n, o)
        }, p(t, o) {
            o & 4 && e !== (e = t[15].sql + "") && Ue(n, e)
        }, d(t) {
            t && G(n)
        }
    }
}

function Ao(r) {
    let e, n;
    return e = new dr({props: {$$slots: {default: [qo]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8388612 && (i.$$scope = {dirty: o, ctx: t}), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Do(r) {
    let e, n;
    return e = new pt({props: {$$slots: {default: [Ao]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8388612 && (i.$$scope = {dirty: o, ctx: t}), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Ho(r) {
    let e, n;
    return e = new ar({props: {id: r[15].id, df: r[15].df}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 4 && (i.id = t[15].id), o & 4 && (i.df = t[15].df), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Mo(r) {
    let e, n;
    return e = new cr({props: {fig: r[15].fig}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 4 && (i.fig = t[15].fig), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Ro(r) {
    let e, n;
    return e = new fr({props: {message: r[15].error}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 4 && (i.message = t[15].error), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Bo(r) {
    let e, n;
    return e = new cr({props: {fig: r[15].fig}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 4 && (i.fig = t[15].fig), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Io(r) {
    let e;
    return {
        c() {
            e = we("结果是否正确？")
        }, m(n, t) {
            U(n, e, t)
        }, d(n) {
            n && G(e)
        }
    }
}

function Nn(r) {
    let e, n, t, o;
    return e = new Ht({props: {message: "是", onSubmit: r[9]}}), t = new Ht({
        props: {
            message: "否",
            onSubmit: r[10]
        }
    }), {
        c() {
            J(e.$$.fragment), n = Z(), J(t.$$.fragment)
        }, m(i, s) {
            F(e, i, s), U(i, n, s), F(t, i, s), o = !0
        }, p(i, s) {
            const l = {};
            s & 1 && (l.onSubmit = i[9]), e.$set(l);
            const c = {};
            s & 1 && (c.onSubmit = i[10]), t.$set(c)
        }, i(i) {
            o || (M(e.$$.fragment, i), M(t.$$.fragment, i), o = !0)
        }, o(i) {
            I(e.$$.fragment, i), I(t.$$.fragment, i), o = !1
        }, d(i) {
            i && G(n), Y(e, i), Y(t, i)
        }
    }
}

function No(r) {
    let e, n, t, o;
    e = new pt({props: {$$slots: {default: [Io]}, $$scope: {ctx: r}}});
    let i = r[0] === null && Nn(r);
    return {
        c() {
            J(e.$$.fragment), n = Z(), i && i.c(), t = nt()
        }, m(s, l) {
            F(e, s, l), U(s, n, l), i && i.m(s, l), U(s, t, l), o = !0
        }, p(s, l) {
            const c = {};
            l & 8388608 && (c.$$scope = {
                dirty: l,
                ctx: s
            }), e.$set(c), s[0] === null ? i ? (i.p(s, l), l & 1 && M(i, 1)) : (i = Nn(s), i.c(), M(i, 1), i.m(t.parentNode, t)) : i && (Ne(), I(i, 1, 1, () => {
                i = null
            }), Qe())
        }, i(s) {
            o || (M(e.$$.fragment, s), M(i), o = !0)
        }, o(s) {
            I(e.$$.fragment, s), I(i), o = !1
        }, d(s) {
            s && (G(n), G(t)), Y(e, s), i && i.d(s)
        }
    }
}

function Qo(r) {
    let e, n;
    return e = new Mt({props: {message: "不，结果不正确."}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Vo(r) {
    let e, n;
    return e = new Mt({props: {message: "是的，结果正确."}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function zo(r) {
    let e, n;
    return e = new ar({props: {id: r[15].id, df: r[15].df}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 4 && (i.id = t[15].id), o & 4 && (i.df = t[15].df), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Qn(r) {
    let e, n;
    return e = new Ht({props: {message: r[18], onSubmit: r[3]}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 4 && (i.message = t[18]), o & 8 && (i.onSubmit = t[3]), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Go(r) {
    let e = r[15].header + "", n, t, o, i, s = _e(r[15].questions), l = [];
    for (let a = 0; a < s.length; a += 1) l[a] = Qn(Mn(r, s, a));
    const c = a => I(l[a], 1, 1, () => {
        l[a] = null
    });
    return {
        c() {
            n = we(e), t = Z();
            for (let a = 0; a < l.length; a += 1) l[a].c();
            o = nt()
        }, m(a, d) {
            U(a, n, d), U(a, t, d);
            for (let g = 0; g < l.length; g += 1) l[g] && l[g].m(a, d);
            U(a, o, d), i = !0
        }, p(a, d) {
            if ((!i || d & 4) && e !== (e = a[15].header + "") && Ue(n, e), d & 12) {
                s = _e(a[15].questions);
                let g;
                for (g = 0; g < s.length; g += 1) {
                    const k = Mn(a, s, g);
                    l[g] ? (l[g].p(k, d), M(l[g], 1)) : (l[g] = Qn(k), l[g].c(), M(l[g], 1), l[g].m(o.parentNode, o))
                }
                for (Ne(), g = s.length; g < l.length; g += 1) c(g);
                Qe()
            }
        }, i(a) {
            if (!i) {
                for (let d = 0; d < s.length; d += 1) M(l[d]);
                i = !0
            }
        }, o(a) {
            l = l.filter(Boolean);
            for (let d = 0; d < l.length; d += 1) I(l[d]);
            i = !1
        }, d(a) {
            a && (G(n), G(t), G(o)), Je(l, a)
        }
    }
}

function Uo(r) {
    let e, n;
    return e = new pt({props: {$$slots: {default: [Go]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8388620 && (i.$$scope = {dirty: o, ctx: t}), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Zo(r) {
    let e, n;
    return e = new Pr({props: {text: r[15].text}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 4 && (i.text = t[15].text), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Wo(r) {
    let e, n;
    return e = new dr({props: {$$slots: {default: [Zo]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8388612 && (i.$$scope = {dirty: o, ctx: t}), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Fo(r) {
    let e, n;
    return e = new pt({props: {$$slots: {default: [Wo]}, $$scope: {ctx: r}}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8388612 && (i.$$scope = {dirty: o, ctx: t}), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Vn(r) {
    let e, n, t, o;
    const i = [Eo, To, Lo, Oo, So, $o, xo, ko, wo], s = [];

    function l(c, a) {
        return c[15].type === "user_question" ? 0 : c[15].type === "sql" ? 1 : c[15].type === "question_list" ? 2 : c[15].type === "df" ? 3 : c[15].type === "plotly_figure" ? 4 : c[15].type === "error" ? 5 : c[15].type === "question_cache" ? 6 : c[15].type === "user_sql" ? 7 : 8
    }

    return e = l(r), n = s[e] = i[e](r), {
        c() {
            n.c(), t = nt()
        }, m(c, a) {
            s[e].m(c, a), U(c, t, a), o = !0
        }, p(c, a) {
            let d = e;
            e = l(c), e === d ? s[e].p(c, a) : (Ne(), I(s[d], 1, 1, () => {
                s[d] = null
            }), Qe(), n = s[e], n ? n.p(c, a) : (n = s[e] = i[e](c), n.c()), M(n, 1), n.m(t.parentNode, t))
        }, i(c) {
            o || (M(n), o = !0)
        }, o(c) {
            I(n), o = !1
        }, d(c) {
            c && G(t), s[e].d(c)
        }
    }
}

function zn(r) {
    let e, n;
    return e = new to({}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Yo(r) {
    let e, n;
    return e = new Ur({props: {onSubmit: r[3]}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8 && (i.onSubmit = t[3]), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function Jo(r) {
    let e, n, t, o;
    e = new ur({props: {message: "新提问", onSubmit: r[5]}});
    let i = _e(r[2]), s = [];
    for (let c = 0; c < i.length; c += 1) s[c] = Un(Dn(r, i, c));
    const l = c => I(s[c], 1, 1, () => {
        s[c] = null
    });
    return {
        c() {
            J(e.$$.fragment), n = Z();
            for (let c = 0; c < s.length; c += 1) s[c].c();
            t = nt()
        }, m(c, a) {
            F(e, c, a), U(c, n, a);
            for (let d = 0; d < s.length; d += 1) s[d] && s[d].m(c, a);
            U(c, t, a), o = !0
        }, p(c, a) {
            const d = {};
            if (a & 32 && (d.onSubmit = c[5]), e.$set(d), a & 20) {
                i = _e(c[2]);
                let g;
                for (g = 0; g < i.length; g += 1) {
                    const k = Dn(c, i, g);
                    s[g] ? (s[g].p(k, a), M(s[g], 1)) : (s[g] = Un(k), s[g].c(), M(s[g], 1), s[g].m(t.parentNode, t))
                }
                for (Ne(), g = i.length; g < s.length; g += 1) l(g);
                Qe()
            }
        }, i(c) {
            if (!o) {
                M(e.$$.fragment, c);
                for (let a = 0; a < i.length; a += 1) M(s[a]);
                o = !0
            }
        }, o(c) {
            I(e.$$.fragment, c), s = s.filter(Boolean);
            for (let a = 0; a < s.length; a += 1) I(s[a]);
            o = !1
        }, d(c) {
            c && (G(n), G(t)), Y(e, c), Je(s, c)
        }
    }
}

function Gn(r) {
    let e, n;

    function t() {
        return r[11](r[12])
    }

    return e = new ur({props: {message: "重新运行 SQL", onSubmit: t}}), {
        c() {
            J(e.$$.fragment)
        }, m(o, i) {
            F(e, o, i), n = !0
        }, p(o, i) {
            r = o;
            const s = {};
            i & 20 && (s.onSubmit = t), e.$set(s)
        }, i(o) {
            n || (M(e.$$.fragment, o), n = !0)
        }, o(o) {
            I(e.$$.fragment, o), n = !1
        }, d(o) {
            Y(e, o)
        }
    }
}

function Un(r) {
    let e, n, t = r[12].type === "question_cache" && Gn(r);
    return {
        c() {
            t && t.c(), e = nt()
        }, m(o, i) {
            t && t.m(o, i), U(o, e, i), n = !0
        }, p(o, i) {
            o[12].type === "question_cache" ? t ? (t.p(o, i), i & 4 && M(t, 1)) : (t = Gn(o), t.c(), M(t, 1), t.m(e.parentNode, e)) : t && (Ne(), I(t, 1, 1, () => {
                t = null
            }), Qe())
        }, i(o) {
            n || (M(t), n = !0)
        }, o(o) {
            I(t), n = !1
        }, d(o) {
            o && G(e), t && t.d(o)
        }
    }
}

function Ko(r) {
    let e, n, t, o, i, s, l, c, a, d, g, k, x, D;
    t = new Rr({});
    let L = r[1] && r[1].type == "question_list" && !r[7] && Bn(r), m = _e(r[2]), h = [];
    for (let u = 0; u < m.length; u += 1) h[u] = Vn(Hn(r, m, u));
    const p = u => I(h[u], 1, 1, () => {
        h[u] = null
    });
    let _ = r[8] && zn();
    d = new Wr({});
    const A = [Jo, Yo], P = [];

    function T(u, f) {
        return u[7] ? 0 : 1
    }

    return k = T(r), x = P[k] = A[k](r), {
        c() {
            e = E("div"), n = E("div"), J(t.$$.fragment), o = Z(), L && L.c(), i = Z(), s = E("ul");
            for (let u = 0; u < h.length; u += 1) h[u].c();
            l = Z(), _ && _.c(), c = Z(), a = E("footer"), J(d.$$.fragment), g = Z(), x.c(), v(s, "class", "mt-16 space-y-5"), v(n, "class", "py-10 lg:py-14"), v(a, "class", "max-w-4xl mx-auto sticky bottom-0 z-10 p-3 sm:py-6"), v(e, "class", "relative h-screen w-full lg:pl-64")
        }, m(u, f) {
            U(u, e, f), w(e, n), F(t, n, null), w(n, o), L && L.m(n, null), w(n, i), w(n, s);
            for (let O = 0; O < h.length; O += 1) h[O] && h[O].m(s, null);
            w(s, l), _ && _.m(s, null), w(e, c), w(e, a), F(d, a, null), w(a, g), P[k].m(a, null), D = !0
        }, p(u, [f]) {
            if (u[1] && u[1].type == "question_list" && !u[7] ? L ? (L.p(u, f), f & 130 && M(L, 1)) : (L = Bn(u), L.c(), M(L, 1), L.m(n, i)) : L && (Ne(), I(L, 1, 1, () => {
                L = null
            }), Qe()), f & 77) {
                m = _e(u[2]);
                let C;
                for (C = 0; C < m.length; C += 1) {
                    const S = Hn(u, m, C);
                    h[C] ? (h[C].p(S, f), M(h[C], 1)) : (h[C] = Vn(S), h[C].c(), M(h[C], 1), h[C].m(s, l))
                }
                for (Ne(), C = m.length; C < h.length; C += 1) p(C);
                Qe()
            }
            u[8] ? _ ? f & 256 && M(_, 1) : (_ = zn(), _.c(), M(_, 1), _.m(s, null)) : _ && (Ne(), I(_, 1, 1, () => {
                _ = null
            }), Qe());
            let O = k;
            k = T(u), k === O ? P[k].p(u, f) : (Ne(), I(P[O], 1, 1, () => {
                P[O] = null
            }), Qe(), x = P[k], x ? x.p(u, f) : (x = P[k] = A[k](u), x.c()), M(x, 1), x.m(a, null))
        }, i(u) {
            if (!D) {
                M(t.$$.fragment, u), M(L);
                for (let f = 0; f < m.length; f += 1) M(h[f]);
                M(_), M(d.$$.fragment, u), M(x), D = !0
            }
        }, o(u) {
            I(t.$$.fragment, u), I(L), h = h.filter(Boolean);
            for (let f = 0; f < h.length; f += 1) I(h[f]);
            I(_), I(d.$$.fragment, u), I(x), D = !1
        }, d(u) {
            u && G(e), Y(t), L && L.d(), Je(h, u), _ && _.d(), Y(d), P[k].d()
        }
    }
}

function Xo(r, e, n) {
    let {suggestedQuestions: t = null} = e, {messageLog: o} = e, {newQuestion: i} = e, {rerunSql: s} = e, {clearMessages: l} = e, {onUpdateSql: c} = e, {question_asked: a} = e, {marked_correct: d} = e, {thinking: g} = e;
    const k = () => n(0, d = !0), x = () => n(0, d = !1), D = L => L.type === "question_cache" ? s(L.id) : void 0;
    return r.$$set = L => {
        "suggestedQuestions" in L && n(1, t = L.suggestedQuestions), "messageLog" in L && n(2, o = L.messageLog), "newQuestion" in L && n(3, i = L.newQuestion), "rerunSql" in L && n(4, s = L.rerunSql), "clearMessages" in L && n(5, l = L.clearMessages), "onUpdateSql" in L && n(6, c = L.onUpdateSql), "question_asked" in L && n(7, a = L.question_asked), "marked_correct" in L && n(0, d = L.marked_correct), "thinking" in L && n(8, g = L.thinking)
    }, [d, t, o, i, s, l, c, a, g, k, x, D]
}

class ei extends Se {
    constructor(e) {
        super(), $e(this, e, Xo, Ko, xe, {
            suggestedQuestions: 1,
            messageLog: 2,
            newQuestion: 3,
            rerunSql: 4,
            clearMessages: 5,
            onUpdateSql: 6,
            question_asked: 7,
            marked_correct: 0,
            thinking: 8
        })
    }
}

function ti(r) {
    let e, n, t, o, i, s, l, c, a, d, g, k, x, D, L, m, h, p, _;
    return {
        c() {
            e = E("div"), n = E("div"), t = E("div"), o = E("div"), i = E("h3"), i.textContent = "确定吗?", s = Z(), l = E("button"), l.innerHTML = '<span class="sr-only">Close</span> <svg class="w-3.5 h-3.5" width="8" height="8" viewBox="0 0 8 8" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M0.258206 1.00652C0.351976 0.912791 0.479126 0.860131 0.611706 0.860131C0.744296 0.860131 0.871447 0.912791 0.965207 1.00652L3.61171 3.65302L6.25822 1.00652C6.30432 0.958771 6.35952 0.920671 6.42052 0.894471C6.48152 0.868271 6.54712 0.854471 6.61352 0.853901C6.67992 0.853321 6.74572 0.865971 6.80722 0.891111C6.86862 0.916251 6.92442 0.953381 6.97142 1.00032C7.01832 1.04727 7.05552 1.1031 7.08062 1.16454C7.10572 1.22599 7.11842 1.29183 7.11782 1.35822C7.11722 1.42461 7.10342 1.49022 7.07722 1.55122C7.05102 1.61222 7.01292 1.6674 6.96522 1.71352L4.31871 4.36002L6.96522 7.00648C7.05632 7.10078 7.10672 7.22708 7.10552 7.35818C7.10442 7.48928 7.05182 7.61468 6.95912 7.70738C6.86642 7.80018 6.74102 7.85268 6.60992 7.85388C6.47882 7.85498 6.35252 7.80458 6.25822 7.71348L3.61171 5.06702L0.965207 7.71348C0.870907 7.80458 0.744606 7.85498 0.613506 7.85388C0.482406 7.85268 0.357007 7.80018 0.264297 7.70738C0.171597 7.61468 0.119017 7.48928 0.117877 7.35818C0.116737 7.22708 0.167126 7.10078 0.258206 7.00648L2.90471 4.36002L0.258206 1.71352C0.164476 1.61976 0.111816 1.4926 0.111816 1.36002C0.111816 1.22744 0.164476 1.10028 0.258206 1.00652Z" fill="currentColor"></path></svg>', c = Z(), a = E("div"), d = E("p"), g = we(r[0]), k = Z(), x = E("div"), D = E("button"), D.textContent = "关闭", L = Z(), m = E("button"), h = we(r[1]), v(i, "class", "font-bold text-gray-800 dark:text-white"), v(l, "type", "button"), v(l, "class", "hs-dropdown-toggle inline-flex flex-shrink-0 justify-center items-center h-8 w-8 rounded-md text-gray-500 hover:text-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-white transition-all text-sm dark:focus:ring-gray-700 dark:focus:ring-offset-gray-800"), v(l, "data-hs-overlay", "#hs-vertically-centered-modal"), v(o, "class", "flex justify-between items-center py-3 px-4 border-b dark:border-gray-700"), v(d, "class", "text-gray-800 dark:text-gray-400"), v(a, "class", "p-4 overflow-y-auto"), v(D, "type", "button"), v(D, "class", "hs-dropdown-toggle py-3 px-4 inline-flex justify-center items-center gap-2 rounded-md border font-medium bg-white text-gray-700 shadow-sm align-middle hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-white focus:ring-blue-600 transition-all text-sm dark:bg-slate-900 dark:hover:bg-slate-800 dark:border-gray-700 dark:text-gray-400 dark:hover:text-white dark:focus:ring-offset-gray-800"), v(D, "data-hs-overlay", "#hs-vertically-centered-modal"), v(m, "class", "py-3 px-4 inline-flex justify-center items-center gap-2 rounded-md border border-transparent font-semibold bg-blue-500 text-white hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all text-sm dark:focus:ring-offset-gray-800"), v(x, "class", "flex justify-end items-center gap-x-2 py-3 px-4 border-t dark:border-gray-700"), v(t, "class", "flex flex-col bg-white border shadow-sm rounded-xl dark:bg-gray-800 dark:border-gray-700 dark:shadow-slate-700/[.7]"), v(n, "class", "hs-overlay-open:mt-7 hs-overlay-open:opacity-100 hs-overlay-open:duration-500 mt-0 opacity-0 ease-out transition-all sm:max-w-lg sm:w-full m-3 sm:mx-auto min-h-[calc(100%-3.5rem)] flex items-center"), v(e, "class", "hs-overlay open w-full h-full fixed top-0 left-0 z-[60] overflow-x-hidden overflow-y-auto")
        }, m(A, P) {
            U(A, e, P), w(e, n), w(n, t), w(t, o), w(o, i), w(o, s), w(o, l), w(t, c), w(t, a), w(a, d), w(d, g), w(t, k), w(t, x), w(x, D), w(x, L), w(x, m), w(m, h), p || (_ = [je(l, "click", function () {
                dt(r[2]) && r[2].apply(this, arguments)
            }), je(D, "click", function () {
                dt(r[2]) && r[2].apply(this, arguments)
            }), je(m, "click", function () {
                dt(r[3]) && r[3].apply(this, arguments)
            })], p = !0)
        }, p(A, [P]) {
            r = A, P & 1 && Ue(g, r[0]), P & 2 && Ue(h, r[1])
        }, i: se, o: se, d(A) {
            A && G(e), p = !1, ot(_)
        }
    }
}

function ni(r, e, n) {
    let {message: t} = e, {buttonLabel: o} = e, {onClose: i} = e, {onConfirm: s} = e;
    return r.$$set = l => {
        "message" in l && n(0, t = l.message), "buttonLabel" in l && n(1, o = l.buttonLabel), "onClose" in l && n(2, i = l.onClose), "onConfirm" in l && n(3, s = l.onConfirm)
    }, [t, o, i, s]
}

class ri extends Se {
    constructor(e) {
        super(), $e(this, e, ni, ti, xe, {message: 0, buttonLabel: 1, onClose: 2, onConfirm: 3})
    }
}

function Zn(r, e, n) {
    const t = r.slice();
    return t[10] = e[n].name, t[11] = e[n].description, t[12] = e[n].example, t
}

function Wn(r) {
    let e, n, t, o, i, s, l, c, a, d, g, k;
    return d = wr(r[7][0]), {
        c() {
            e = E("div"), n = E("div"), t = E("input"), o = Z(), i = E("label"), s = E("span"), s.textContent = `${r[10]}`, l = Z(), c = E("span"), c.textContent = `${r[11]}`, a = Z(), v(t, "id", "hs-radio-" + r[10]), t.__value = r[10], bt(t, t.__value), v(t, "name", "hs-radio-with-description"), v(t, "type", "radio"), v(t, "class", "border-gray-200 rounded-full text-blue-600 focus:ring-blue-500 dark:bg-gray-800 dark:border-gray-700 dark:checked:bg-blue-500 dark:checked:border-blue-500 dark:focus:ring-offset-gray-800"), v(t, "aria-describedby", "hs-radio-delete-description"), v(n, "class", "flex items-center h-5 mt-1"), v(s, "class", "block text-sm font-semibold text-gray-800 dark:text-gray-300"), v(c, "id", "hs-radio-ddl-description"), v(c, "class", "block text-sm text-gray-600 dark:text-gray-500"), v(i, "for", "hs-radio-" + r[10]), v(i, "class", "ml-3"), v(e, "class", "relative flex items-start"), d.p(t)
        }, m(x, D) {
            U(x, e, D), w(e, n), w(n, t), t.checked = t.__value === r[0], w(e, o), w(e, i), w(i, s), w(i, l), w(i, c), w(e, a), g || (k = je(t, "change", r[6]), g = !0)
        }, p(x, D) {
            D & 1 && (t.checked = t.__value === x[0])
        }, d(x) {
            x && G(e), d.r(), g = !1, k()
        }
    }
}

function oi(r) {
    let e, n, t, o, i, s, l, c, a, d, g, k, x, D, L, m, h, p, _, A, P, T, u, f, O, C = _e(r[3]), S = [];
    for (let j = 0; j < C.length; j += 1) S[j] = Wn(Zn(r, C, j));
    return {
        c() {
            var j;
            e = E("div"), n = E("div"), t = E("div"), o = E("div"), i = E("h2"), i.textContent = "添加训练数据", s = Z(), l = E("button"), l.innerHTML = '<span class="sr-only">Close</span> <svg class="w-3.5 h-3.5" width="8" height="8" viewBox="0 0 8 8" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M0.258206 1.00652C0.351976 0.912791 0.479126 0.860131 0.611706 0.860131C0.744296 0.860131 0.871447 0.912791 0.965207 1.00652L3.61171 3.65302L6.25822 1.00652C6.30432 0.958771 6.35952 0.920671 6.42052 0.894471C6.48152 0.868271 6.54712 0.854471 6.61352 0.853901C6.67992 0.853321 6.74572 0.865971 6.80722 0.891111C6.86862 0.916251 6.92442 0.953381 6.97142 1.00032C7.01832 1.04727 7.05552 1.1031 7.08062 1.16454C7.10572 1.22599 7.11842 1.29183 7.11782 1.35822C7.11722 1.42461 7.10342 1.49022 7.07722 1.55122C7.05102 1.61222 7.01292 1.6674 6.96522 1.71352L4.31871 4.36002L6.96522 7.00648C7.05632 7.10078 7.10672 7.22708 7.10552 7.35818C7.10442 7.48928 7.05182 7.61468 6.95912 7.70738C6.86642 7.80018 6.74102 7.85268 6.60992 7.85388C6.47882 7.85498 6.35252 7.80458 6.25822 7.71348L3.61171 5.06702L0.965207 7.71348C0.870907 7.80458 0.744606 7.85498 0.613506 7.85388C0.482406 7.85268 0.357007 7.80018 0.264297 7.70738C0.171597 7.61468 0.119017 7.48928 0.117877 7.35818C0.116737 7.22708 0.167126 7.10078 0.258206 7.00648L2.90471 4.36002L0.258206 1.71352C0.164476 1.61976 0.111816 1.4926 0.111816 1.36002C0.111816 1.22744 0.164476 1.10028 0.258206 1.00652Z" fill="currentColor"></path></svg>', c = Z(), a = E("span"), a.textContent = "数据训练的类型", d = Z(), g = E("div");
            for (let R = 0; R < S.length; R += 1) S[R].c();
            k = Z(), x = E("div"), D = E("label"), L = we("你的 "), m = we(r[0]), h = Z(), p = E("div"), _ = E("textarea"), P = Z(), T = E("div"), u = E("button"), u.textContent = "保存", v(i, "class", "text-xl text-gray-800 font-bold sm:text-3xl dark:text-white"), v(l, "type", "button"), v(l, "class", "hs-dropdown-toggle inline-flex flex-shrink-0 justify-center items-center h-8 w-8 rounded-md text-gray-500 hover:text-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-white transition-all text-sm dark:focus:ring-gray-700 dark:focus:ring-offset-gray-800"), v(l, "data-hs-overlay", "#hs-vertically-centered-modal"), v(o, "class", "flex justify-between items-center py-3 px-4 border-b dark:border-gray-700 mb-2"), v(a, "class", "block mb-2 text-sm font-medium dark:text-white"), v(g, "class", "grid space-y-3 mb-1"), v(D, "for", "hs-feedback-post-comment-textarea-1"), v(D, "class", "block mt-2 mb-2 text-sm font-medium dark:text-white"), v(_, "id", "hs-feedback-post-comment-textarea-1"), v(_, "name", "hs-feedback-post-comment-textarea-1"), v(_, "rows", "3"), v(_, "class", "py-3 px-4 block w-full border border-gray-200 rounded-md text-sm focus:border-blue-500 focus:ring-blue-500 sm:p-4 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400"), v(_, "placeholder", A = ((j = r[3].find(r[8])) == null ? void 0 : j.example) ?? "没有可用的例子"), v(p, "class", "mt-1"), v(x, "class", "mt-2 border-t dark:border-gray-700"), v(u, "class", "py-3 px-4 inline-flex justify-center items-center gap-2 rounded-md border border-transparent font-semibold bg-blue-500 text-white hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all dark:focus:ring-offset-gray-800"), v(T, "class", "mt-6 grid"), v(t, "class", "mt-5 p-4 relative z-10 bg-white border rounded-xl sm:mt-10 md:p-10 dark:bg-gray-800 dark:border-gray-700"), v(n, "class", "mx-auto max-w-2xl"), v(e, "class", "max-w-[85rem] px-4 py-10 sm:px-6 lg:px-8 lg:py-14 mx-auto")
        }, m(j, R) {
            U(j, e, R), w(e, n), w(n, t), w(t, o), w(o, i), w(o, s), w(o, l), w(t, c), w(t, a), w(t, d), w(t, g);
            for (let q = 0; q < S.length; q += 1) S[q] && S[q].m(g, null);
            w(t, k), w(t, x), w(x, D), w(D, L), w(D, m), w(x, h), w(x, p), w(p, _), bt(_, r[2]), w(t, P), w(t, T), w(T, u), f || (O = [je(l, "click", function () {
                dt(r[1]) && r[1].apply(this, arguments)
            }), je(_, "input", r[9]), je(u, "click", r[4])], f = !0)
        }, p(j, [R]) {
            var q;
            if (r = j, R & 9) {
                C = _e(r[3]);
                let B;
                for (B = 0; B < C.length; B += 1) {
                    const N = Zn(r, C, B);
                    S[B] ? S[B].p(N, R) : (S[B] = Wn(N), S[B].c(), S[B].m(g, null))
                }
                for (; B < S.length; B += 1) S[B].d(1);
                S.length = C.length
            }
            R & 1 && Ue(m, r[0]), R & 1 && A !== (A = ((q = r[3].find(r[8])) == null ? void 0 : q.example) ?? "No example available") && v(_, "placeholder", A), R & 4 && bt(_, r[2])
        }, i: se, o: se, d(j) {
            j && G(e), Je(S, j), f = !1, ot(O)
        }
    }
}

function ii(r, e, n) {
    let {onDismiss: t} = e, {onTrain: o} = e, {selectedTrainingDataType: i = "SQL"} = e, s = [{
        name: "DDL",
        description: "问题与答案以英文逗号分割:  这是定义数据库表结构的语句.问题与答案以英文逗号分割",
        example: "表的中文含义, CREATE TABLE table_name (column_1 datatype, column_2 datatype, column_3 datatype);"
    }, {
        name: "Documentation",
        description: "这可以是任何基于文本的文档。保持篇幅小，并专注于单一主题。",
        example: " 我们对 ABC 的定义是 XYZ."
    }, {
        name: "SQL",
        description: " 问题与答案以英文逗号分割这可以是任何有效的 SQL 语句。越多越好。",
        example: "SQL问句?, SELECT column_1, column_2 FROM table_name;"
    }], l = "";
    const c = () => {
        o(l, i.toLowerCase())
    }, a = [[]];

    function d() {
        i = this.__value, n(0, i)
    }

    const g = x => x.name === i;

    function k() {
        l = this.value, n(2, l)
    }

    return r.$$set = x => {
        "onDismiss" in x && n(1, t = x.onDismiss), "onTrain" in x && n(5, o = x.onTrain), "selectedTrainingDataType" in x && n(0, i = x.selectedTrainingDataType)
    }, [i, t, l, s, c, o, d, a, g, k]
}

class si extends Se {
    constructor(e) {
        super(), $e(this, e, ii, oi, xe, {onDismiss: 1, onTrain: 5, selectedTrainingDataType: 0})
    }
}

function Fn(r, e, n) {
    const t = r.slice();
    return t[21] = e[n], t
}

function Yn(r, e, n) {
    const t = r.slice();
    return t[24] = e[n], t
}

function Jn(r, e, n) {
    const t = r.slice();
    return t[24] = e[n], t
}

function Kn(r) {
    let e, n;
    return e = new si({props: {onDismiss: r[13], onTrain: r[0]}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 1 && (i.onTrain = t[0]), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function li(r) {
    let e;
    return {
        c() {
            e = we("操作")
        }, m(n, t) {
            U(n, e, t)
        }, p: se, d(n) {
            n && G(e)
        }
    }
}

function ai(r) {
    let e = r[24] + "", n;
    return {
        c() {
            n = we(e)
        }, m(t, o) {
            U(t, n, o)
        }, p: se, d(t) {
            t && G(n)
        }
    }
}

function Xn(r) {
    let e, n, t, o;

    function i(c, a) {
        return c[24] != "id" ? ai : li
    }

    let l = i(r)(r);
    return {
        c() {
            e = E("th"), n = E("div"), t = E("span"), l.c(), o = Z(), v(t, "class", "text-xs font-semibold uppercase tracking-wide text-gray-800 dark:text-gray-200"), v(n, "class", "flex items-center gap-x-2"), v(e, "scope", "col"), v(e, "class", "px-6 py-3 text-left")
        }, m(c, a) {
            U(c, e, a), w(e, n), w(n, t), l.m(t, null), w(e, o)
        }, p(c, a) {
            l.p(c, a)
        }, d(c) {
            c && G(e), l.d()
        }
    }
}

function ci(r) {
    let e, n, t;

    function o() {
        return r[18](r[21], r[24])
    }

    return {
        c() {
            e = E("button"), e.textContent = "删除", v(e, "type", "button"), v(e, "class", "py-2 px-3 inline-flex justify-center items-center gap-2 rounded-md border-2 border-red-200 font-semibold text-red-500 hover:text-white hover:bg-red-500 hover:border-red-500 focus:outline-none focus:ring-2 focus:ring-red-200 focus:ring-offset-2 transition-all text-sm dark:focus:ring-offset-gray-800")
        }, m(i, s) {
            U(i, e, s), n || (t = je(e, "click", o), n = !0)
        }, p(i, s) {
            r = i
        }, d(i) {
            i && G(e), n = !1, t()
        }
    }
}

function ui(r) {
    let e, n = r[21][r[24]] + "", t;
    return {
        c() {
            e = E("span"), t = we(n), v(e, "class", "text-gray-800 dark:text-gray-200")
        }, m(o, i) {
            U(o, e, i), w(e, t)
        }, p(o, i) {
            i & 16 && n !== (n = o[21][o[24]] + "") && Ue(t, n)
        }, d(o) {
            o && G(e)
        }
    }
}

function er(r) {
    let e, n;

    function t(s, l) {
        return s[24] != "id" ? ui : ci
    }

    let i = t(r)(r);
    return {
        c() {
            e = E("td"), n = E("div"), i.c(), v(n, "class", "px-6 py-3"), v(e, "class", "h-px w-px ")
        }, m(s, l) {
            U(s, e, l), w(e, n), i.m(n, null)
        }, p(s, l) {
            i.p(s, l)
        }, d(s) {
            s && G(e), i.d()
        }
    }
}

function tr(r) {
    let e, n, t = _e(r[8]), o = [];
    for (let i = 0; i < t.length; i += 1) o[i] = er(Yn(r, t, i));
    return {
        c() {
            e = E("tr");
            for (let i = 0; i < o.length; i += 1) o[i].c();
            n = Z()
        }, m(i, s) {
            U(i, e, s);
            for (let l = 0; l < o.length; l += 1) o[l] && o[l].m(e, null);
            w(e, n)
        }, p(i, s) {
            if (s & 304) {
                t = _e(i[8]);
                let l;
                for (l = 0; l < t.length; l += 1) {
                    const c = Yn(i, t, l);
                    o[l] ? o[l].p(c, s) : (o[l] = er(c), o[l].c(), o[l].m(e, n))
                }
                for (; l < o.length; l += 1) o[l].d(1);
                o.length = t.length
            }
        }, d(i) {
            i && G(e), Je(o, i)
        }
    }
}

function nr(r) {
    let e, n;
    return e = new ri({
        props: {
            message: "您确定要删除此项吗?",
            buttonLabel: "删除",
            onClose: r[19],
            onConfirm: r[20]
        }
    }), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 32 && (i.onClose = t[19]), o & 34 && (i.onConfirm = t[20]), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function fi(r) {
    let e, n, t, o, i, s, l, c, a, d, g, k, x, D, L, m, h, p, _, A, P, T, u, f, O, C, S, j = r[2] + 1 + "", R, q,
        B = Math.min(r[3], r[7].length) + "", N, ie, re, Ye, Ae, De, Ve, lt, me, it, ze, at, Lt, Tt, Te = r[6] && Kn(r),
        ct = _e(r[8]), Pe = [];
    for (let X = 0; X < ct.length; X += 1) Pe[X] = Xn(Jn(r, ct, X));
    let Be = _e(r[4]), Oe = [];
    for (let X = 0; X < Be.length; X += 1) Oe[X] = tr(Fn(r, Be, X));
    let Le = r[5] != null && nr(r);
    return {
        c() {
            Te && Te.c(), e = Z(), n = E("div"), t = E("div"), o = E("div"), i = E("div"), s = E("div"), l = E("div"), c = E("div"), c.innerHTML = '<h2 class="text-xl font-semibold text-gray-800 dark:text-gray-200">训练数据</h2> <p class="text-sm text-gray-600 dark:text-gray-400">添加或者移除训练数据. 良好的训练数据是准确性的关键.</p>', a = Z(), d = E("div"), g = E("div"), k = E("button"), k.textContent = "查看所有", x = Z(), D = E("button"), D.innerHTML = `<svg class="w-3 h-3" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M2.63452 7.50001L13.6345 7.5M8.13452 13V2" stroke="currentColor" stroke-width="2" stroke-linecap="round"></path></svg>
                    添加训练数据`, L = Z(), m = E("table"), h = E("thead"), p = E("tr");
            for (let X = 0; X < Pe.length; X += 1) Pe[X].c();
            _ = Z(), A = E("tbody");
            for (let X = 0; X < Oe.length; X += 1) Oe[X].c();
            P = Z(), T = E("div"), u = E("div"), f = E("p"), f.textContent = "显示:", O = Z(), C = E("div"), S = E("span"), R = we(j), q = we(" - "), N = we(B), ie = Z(), re = E("p"), re.textContent = `of ${r[7].length}`, Ye = Z(), Ae = E("div"), De = E("div"), Ve = E("button"), Ve.innerHTML = `<svg class="w-3 h-3" xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M11.354 1.646a.5.5 0 0 1 0 .708L5.707 8l5.647 5.646a.5.5 0 0 1-.708.708l-6-6a.5.5 0 0 1 0-.708l6-6a.5.5 0 0 1 .708 0z"></path></svg>
                    上一页`, lt = Z(), me = E("button"), me.innerHTML = `下一页
                    <svg class="w-3 h-3" xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path fill-rule="evenodd" d="M4.646 1.646a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1 0 .708l-6 6a.5.5 0 0 1-.708-.708L10.293 8 4.646 2.354a.5.5 0 0 1 0-.708z"></path></svg>`, it = Z(), Le && Le.c(), ze = nt(), v(k, "class", "py-2 px-3 inline-flex justify-center items-center gap-2 rounded-md border font-medium bg-white text-gray-700 shadow-sm align-middle hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-white focus:ring-blue-600 transition-all text-sm dark:bg-slate-900 dark:hover:bg-slate-800 dark:border-gray-700 dark:text-gray-400 dark:hover:text-white dark:focus:ring-offset-gray-800"), v(D, "class", "py-2 px-3 inline-flex justify-center items-center gap-2 rounded-md border border-transparent font-semibold bg-blue-500 text-white hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all text-sm dark:focus:ring-offset-gray-800"), v(g, "class", "inline-flex gap-x-2"), v(l, "class", "px-6 py-4 grid gap-3 md:flex md:justify-between md:items-center border-b border-gray-200 dark:border-gray-700"), v(h, "class", "bg-gray-50 dark:bg-slate-800"), v(A, "class", "divide-y divide-gray-200 dark:divide-gray-700"), v(m, "class", "min-w-full divide-y divide-gray-200 dark:divide-gray-700"), v(f, "class", "text-sm text-gray-600 dark:text-gray-400"), v(S, "class", "py-2 px-3 pr-9 block w-full border-gray-200 rounded-md text-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-slate-900 dark:border-gray-700 dark:text-gray-400"), v(C, "class", "max-w-sm space-y-3"), v(re, "class", "text-sm text-gray-600 dark:text-gray-400"), v(u, "class", "inline-flex items-center gap-x-2"), v(Ve, "type", "button"), v(Ve, "class", "py-2 px-3 inline-flex justify-center items-center gap-2 rounded-md border font-medium bg-white text-gray-700 shadow-sm align-middle hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-white focus:ring-blue-600 transition-all text-sm dark:bg-slate-900 dark:hover:bg-slate-800 dark:border-gray-700 dark:text-gray-400 dark:hover:text-white dark:focus:ring-offset-gray-800"), v(me, "type", "button"), v(me, "class", "py-2 px-3 inline-flex justify-center items-center gap-2 rounded-md border font-medium bg-white text-gray-700 shadow-sm align-middle hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-white focus:ring-blue-600 transition-all text-sm dark:bg-slate-900 dark:hover:bg-slate-800 dark:border-gray-700 dark:text-gray-400 dark:hover:text-white dark:focus:ring-offset-gray-800"), v(De, "class", "inline-flex gap-x-2"), v(T, "class", "px-6 py-4 grid gap-3 md:flex md:justify-between md:items-center border-t border-gray-200 dark:border-gray-700"), v(s, "class", "bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden dark:bg-slate-900 dark:border-gray-700"), v(i, "class", "p-1.5 min-w-full inline-block align-middle"), v(o, "class", "-m-1.5 overflow-x-auto"), v(t, "class", "flex flex-col"), v(n, "class", "max-w-[85rem] px-4 py-10 sm:px-6 lg:px-8 lg:py-14 mx-auto")
        }, m(X, He) {
            Te && Te.m(X, He), U(X, e, He), U(X, n, He), w(n, t), w(t, o), w(o, i), w(i, s), w(s, l), w(l, c), w(l, a), w(l, d), w(d, g), w(g, k), w(g, x), w(g, D), w(s, L), w(s, m), w(m, h), w(h, p);
            for (let oe = 0; oe < Pe.length; oe += 1) Pe[oe] && Pe[oe].m(p, null);
            w(m, _), w(m, A);
            for (let oe = 0; oe < Oe.length; oe += 1) Oe[oe] && Oe[oe].m(A, null);
            w(s, P), w(s, T), w(T, u), w(u, f), w(u, O), w(u, C), w(C, S), w(S, R), w(S, q), w(S, N), w(u, ie), w(u, re), w(T, Ye), w(T, Ae), w(Ae, De), w(De, Ve), w(De, lt), w(De, me), U(X, it, He), Le && Le.m(X, He), U(X, ze, He), at = !0, Lt || (Tt = [je(k, "click", r[11]), je(D, "click", r[12]), je(Ve, "click", r[9]), je(me, "click", r[10])], Lt = !0)
        }, p(X, [He]) {
            if (X[6] ? Te ? (Te.p(X, He), He & 64 && M(Te, 1)) : (Te = Kn(X), Te.c(), M(Te, 1), Te.m(e.parentNode, e)) : Te && (Ne(), I(Te, 1, 1, () => {
                Te = null
            }), Qe()), He & 256) {
                ct = _e(X[8]);
                let oe;
                for (oe = 0; oe < ct.length; oe += 1) {
                    const gt = Jn(X, ct, oe);
                    Pe[oe] ? Pe[oe].p(gt, He) : (Pe[oe] = Xn(gt), Pe[oe].c(), Pe[oe].m(p, null))
                }
                for (; oe < Pe.length; oe += 1) Pe[oe].d(1);
                Pe.length = ct.length
            }
            if (He & 304) {
                Be = _e(X[4]);
                let oe;
                for (oe = 0; oe < Be.length; oe += 1) {
                    const gt = Fn(X, Be, oe);
                    Oe[oe] ? Oe[oe].p(gt, He) : (Oe[oe] = tr(gt), Oe[oe].c(), Oe[oe].m(A, null))
                }
                for (; oe < Oe.length; oe += 1) Oe[oe].d(1);
                Oe.length = Be.length
            }
            (!at || He & 4) && j !== (j = X[2] + 1 + "") && Ue(R, j), (!at || He & 8) && B !== (B = Math.min(X[3], X[7].length) + "") && Ue(N, B), X[5] != null ? Le ? (Le.p(X, He), He & 32 && M(Le, 1)) : (Le = nr(X), Le.c(), M(Le, 1), Le.m(ze.parentNode, ze)) : Le && (Ne(), I(Le, 1, 1, () => {
                Le = null
            }), Qe())
        }, i(X) {
            at || (M(Te), M(Le), at = !0)
        }, o(X) {
            I(Te), I(Le), at = !1
        }, d(X) {
            X && (G(e), G(n), G(it), G(ze)), Te && Te.d(X), Je(Pe, X), Je(Oe, X), Le && Le.d(X), Lt = !1, ot(Tt)
        }
    }
}

function di(r, e, n) {
    let {df: t} = e, {onTrain: o} = e, {removeTrainingData: i} = e, s = JSON.parse(t),
        l = s.length > 0 ? Object.keys(s[0]) : [], c = 10, a = 1, d = Math.ceil(s.length / c), g = (a - 1) * c,
        k = a * c, x = s.slice(g, k);
    const D = () => {
        a > 1 && n(16, a--, a)
    }, L = () => {
        a < d && n(16, a++, a)
    }, m = () => {
        n(16, a = 1), n(15, c = s.length)
    };
    let h = null, p = !1;
    const _ = () => {
        n(6, p = !0)
    }, A = () => {
        n(6, p = !1)
    }, P = (f, O) => {
        n(5, h = f[O])
    }, T = () => {
        n(5, h = null)
    }, u = () => {
        h && i(h)
    };
    return r.$$set = f => {
        "df" in f && n(14, t = f.df), "onTrain" in f && n(0, o = f.onTrain), "removeTrainingData" in f && n(1, i = f.removeTrainingData)
    }, r.$$.update = () => {
        r.$$.dirty & 98304 && n(2, g = (a - 1) * c), r.$$.dirty & 98304 && n(3, k = a * c), r.$$.dirty & 12 && n(4, x = s.slice(g, k)), r.$$.dirty & 32768 && n(17, d = Math.ceil(s.length / c)), r.$$.dirty & 196608 && console.log(a, d)
    }, [o, i, g, k, x, h, p, s, l, D, L, m, _, A, t, c, a, d, P, T, u]
}

class pi extends Se {
    constructor(e) {
        super(), $e(this, e, di, fi, xe, {df: 14, onTrain: 0, removeTrainingData: 1})
    }
}

function gi(r) {
    let e;
    return {
        c() {
            e = E("div"), e.innerHTML = '<div class="flex flex-auto flex-col justify-center items-center p-4 md:p-5"><div class="flex justify-center"><div class="animate-spin inline-block w-6 h-6 border-[3px] border-current border-t-transparent text-blue-600 rounded-full" role="status" aria-label="loading"><span class="sr-only">Loading...</span></div></div></div>', v(e, "class", "min-h-[15rem] flex flex-col bg-white border shadow-sm rounded-xl dark:bg-gray-800 dark:border-gray-700 dark:shadow-slate-700/[.7]")
        }, m(n, t) {
            U(n, e, t)
        }, p: se, i: se, o: se, d(n) {
            n && G(e)
        }
    }
}

function hi(r) {
    let e, n, t, o;
    const i = [yi, mi], s = [];

    function l(c, a) {
        return c[0].type === "df" ? 0 : c[0].type === "error" ? 1 : -1
    }

    return ~(e = l(r)) && (n = s[e] = i[e](r)), {
        c() {
            n && n.c(), t = nt()
        }, m(c, a) {
            ~e && s[e].m(c, a), U(c, t, a), o = !0
        }, p(c, a) {
            let d = e;
            e = l(c), e === d ? ~e && s[e].p(c, a) : (n && (Ne(), I(s[d], 1, 1, () => {
                s[d] = null
            }), Qe()), ~e ? (n = s[e], n ? n.p(c, a) : (n = s[e] = i[e](c), n.c()), M(n, 1), n.m(t.parentNode, t)) : n = null)
        }, i(c) {
            o || (M(n), o = !0)
        }, o(c) {
            I(n), o = !1
        }, d(c) {
            c && G(t), ~e && s[e].d(c)
        }
    }
}

function mi(r) {
    let e, n;
    return e = new fr({props: {message: r[0].error}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 1 && (i.message = t[0].error), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function yi(r) {
    let e, n;
    return e = new pi({props: {df: r[0].df, removeTrainingData: r[1], onTrain: r[2]}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 1 && (i.df = t[0].df), o & 2 && (i.removeTrainingData = t[1]), o & 4 && (i.onTrain = t[2]), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function bi(r) {
    let e, n, t, o, i;
    const s = [hi, gi], l = [];

    function c(a, d) {
        return a[0] !== null ? 0 : 1
    }

    return t = c(r), o = l[t] = s[t](r), {
        c() {
            e = E("div"), n = E("div"), o.c(), v(n, "class", "py-10 lg:py-14"), v(e, "class", "relative h-screen w-full lg:pl-64")
        }, m(a, d) {
            U(a, e, d), w(e, n), l[t].m(n, null), i = !0
        }, p(a, [d]) {
            let g = t;
            t = c(a), t === g ? l[t].p(a, d) : (Ne(), I(l[g], 1, 1, () => {
                l[g] = null
            }), Qe(), o = l[t], o ? o.p(a, d) : (o = l[t] = s[t](a), o.c()), M(o, 1), o.m(n, null))
        }, i(a) {
            i || (M(o), i = !0)
        }, o(a) {
            I(o), i = !1
        }, d(a) {
            a && G(e), l[t].d()
        }
    }
}

function vi(r, e, n) {
    let {trainingData: t} = e, {removeTrainingData: o} = e, {onTrain: i} = e;
    return r.$$set = s => {
        "trainingData" in s && n(0, t = s.trainingData), "removeTrainingData" in s && n(1, o = s.removeTrainingData), "onTrain" in s && n(2, i = s.onTrain)
    }, [t, o, i]
}

class _i extends Se {
    constructor(e) {
        super(), $e(this, e, vi, bi, xe, {trainingData: 0, removeTrainingData: 1, onTrain: 2})
    }
}

function wi(r) {
    let e, n;
    return e = new _i({props: {trainingData: r[3], removeTrainingData: r[14], onTrain: r[15]}}), {
        c() {
            J(e.$$.fragment)
        }, m(t, o) {
            F(e, t, o), n = !0
        }, p(t, o) {
            const i = {};
            o & 8 && (i.trainingData = t[3]), e.$set(i)
        }, i(t) {
            n || (M(e.$$.fragment, t), n = !0)
        }, o(t) {
            I(e.$$.fragment, t), n = !1
        }, d(t) {
            Y(e, t)
        }
    }
}

function ki(r) {
    let e, n, t, o, i;

    function s(d) {
        r[17](d)
    }

    function l(d) {
        r[18](d)
    }

    function c(d) {
        r[19](d)
    }

    let a = {
        suggestedQuestions: r[2],
        messageLog: r[1],
        newQuestion: r[9],
        rerunSql: r[10],
        clearMessages: r[8],
        onUpdateSql: r[16]
    };
    return r[4] !== void 0 && (a.question_asked = r[4]), r[5] !== void 0 && (a.thinking = r[5]), r[0] !== void 0 && (a.marked_correct = r[0]), e = new ei({props: a}), At.push(() => on(e, "question_asked", s)), At.push(() => on(e, "thinking", l)), At.push(() => on(e, "marked_correct", c)), {
        c() {
            J(e.$$.fragment)
        }, m(d, g) {
            F(e, d, g), i = !0
        }, p(d, g) {
            const k = {};
            g & 4 && (k.suggestedQuestions = d[2]), g & 2 && (k.messageLog = d[1]), !n && g & 16 && (n = !0, k.question_asked = d[4], nn(() => n = !1)), !t && g & 32 && (t = !0, k.thinking = d[5], nn(() => t = !1)), !o && g & 1 && (o = !0, k.marked_correct = d[0], nn(() => o = !1)), e.$set(k)
        }, i(d) {
            i || (M(e.$$.fragment, d), i = !0)
        }, o(d) {
            I(e.$$.fragment, d), i = !1
        }, d(d) {
            Y(e, d)
        }
    }
}

function xi(r) {
    let e, n, t, o, i, s;
    n = new Dr({
        props: {
            getTrainingData: r[11],
            newQuestionPage: r[12],
            loadQuestionPage: r[13],
            questionHistory: r[7]
        }
    });
    const l = [ki, wi], c = [];

    function a(d, g) {
        return d[6] === "chat" ? 0 : d[6] === "training-data" ? 1 : -1
    }

    return ~(o = a(r)) && (i = c[o] = l[o](r)), {
        c() {
            e = E("main"), J(n.$$.fragment), t = Z(), i && i.c()
        }, m(d, g) {
            U(d, e, g), F(n, e, null), w(e, t), ~o && c[o].m(e, null), s = !0
        }, p(d, [g]) {
            const k = {};
            g & 128 && (k.questionHistory = d[7]), n.$set(k);
            let x = o;
            o = a(d), o === x ? ~o && c[o].p(d, g) : (i && (Ne(), I(c[x], 1, 1, () => {
                c[x] = null
            }), Qe()), ~o ? (i = c[o], i ? i.p(d, g) : (i = c[o] = l[o](d), i.c()), M(i, 1), i.m(e, null)) : i = null)
        }, i(d) {
            s || (M(n.$$.fragment, d), M(i), s = !0)
        }, o(d) {
            I(n.$$.fragment, d), I(i), s = !1
        }, d(d) {
            d && G(e), Y(n), ~o && c[o].d()
        }
    }
}

function $i() {
    setTimeout(() => {
        window.scrollTo({top: document.body.scrollHeight, behavior: "smooth"})
    }, 100)
}

function Si(r, e, n) {
    sr(async () => {
        D(), new URL(window.location.href).hash.slice(1) === "training-data" ? L() : m()
    });
    let t = [], o = null, i = null, s = !1, l = !1, c = null, a, d = [];

    function g() {
        n(1, t = []), n(4, s = !1), n(5, l = !1), n(0, c = null)
    }

    function k(q) {
        g(), _({
            type: "user_question",
            question: q
        }), n(4, s = !0), f("generate_sql", "GET", {question: q}).then(_).then(B => {
            B.type === "sql" && (window.location.hash = B.id, f("run_sql", "GET", {id: B.id}).then(_).then(N =>
                {
                // 画图
                N.type === "df" && f("generate_plotly_figure", "GET", {id: N.id}).then(_).then(ie => {
                    ie.type === "plotly_figure" && (n(7, d = [...d, {
                        question: q,
                        id: ie.id
                    }]), f("generate_followup_questions", "GET", {id: ie.id}).then(_))
                })
            }

            ))
        })
    }

    function x(q) {
        _({type: "user_question", question: "重新运行SQL"}), f("run_sql", "GET", {id: q}).then(_).then(B => {
            // 画图
            B.type === "df" && f("generate_plotly_figure", "GET", {id: B.id}).then(_).then(N => {
                N.type === "plotly_figure" && f("generate_followup_questions", "GET", {id: N.id}).then(_)
            })

        })
    }

    function D() {
        f("get_question_history", "GET", []).then(T)
    }

    function L() {
        window.location.hash = "training-data", n(6, a = "training-data"), f("get_training_data", "GET", []).then(A)
    }

    function m() {
        window.location.hash = "", n(6, a = "chat"), g(), o || f("generate_questions", "GET", []).then(P)
    }

    function h(q) {
        window.location.hash = q, n(6, a = "chat"), g(), n(4, s = !0), f("load_question", "GET", {id: q}).then(_)
    }

    function p(q) {
        n(3, i = null), f("remove_training_data", "POST", {id: q}).then(B => {
            f("get_training_data", "GET", []).then(A)
        })
    }

    function _(q) {
        return n(1, t = [...t, q]), $i(), q
    }

    function A(q) {
        return n(3, i = q), q
    }

    function P(q) {
        return n(2, o = q), q
    }

    function T(q) {
        return q.type === "question_history" && n(7, d = q.questions), q
    }

    function u(q, B) {
        n(3, i = null);
        let N = {};
        N[B] = q, f("train", "POST", N).then(A).then(ie => {
            ie.type !== "error" && f("get_training_data", "GET", []).then(A)
        })
    }

    async function f(q, B, N) {
        try {
            n(5, l = !0);
            let ie = "", re;
            if (B === "GET") ie = Object.entries(N).filter(([Ae, De]) => Ae !== "endpoint" && Ae !== "addMessage").map(([Ae, De]) => `${encodeURIComponent(Ae)}=${encodeURIComponent(De)}`).join("&"), re = await fetch(`/api/v0/${q}?${ie}`); else {
                let Ae = JSON.stringify(N);
                re = await fetch(`/api/v0/${q}`, {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: Ae
                })
            }
            if (!re.ok) throw new Error("服务器返回错误。请参阅服务器日志了解更多详细信息.");
            const Ye = await re.json();
            return n(5, l = !1), Ye
        } catch (ie) {
            return n(5, l = !1), {type: "error", error: String(ie)}
        }
    }

    function O() {
        let q = t.find(B => B.type === "user_question");
        if (q && q.type === "user_question") {
            let B = t.find(N => N.type === "sql");
            if (B && B.type === "sql") return {question: q.question, sql: B.text}
        }
        return null
    }

    function C(q) {
        let B = t.find(N => N.type === "user_question");
        if (B && B.type === "user_question") {
            let N = {question: B.question, sql: q};
            f("train", "POST", N), n(1, t = t.filter(ie => ie.type !== "user_sql")), _({
                type: "sql",
                text: q,
                id: window.location.hash
            })
        }
    }

    function S(q) {
        s = q, n(4, s)
    }

    function j(q) {
        l = q, n(5, l)
    }

    function R(q) {
        c = q, n(0, c)
    }

    return r.$$.update = () => {
        if (r.$$.dirty & 1) if (c === !0) {
            let q = O();
            q && f("train", "POST", q)
        } else c === !1 && _({type: "user_sql"})
    }, [c, t, o, i, s, l, a, d, g, k, x, L, m, h, p, u, C, S, j, R]
}

class Oi extends Se {
    constructor(e) {
        super(), $e(this, e, Si, xi, xe, {})
    }
}

new Oi({target: document.getElementById("app")});
