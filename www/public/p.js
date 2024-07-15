/* eslint-disable */
!(function () {
  "use strict";
  var a = window.location,
    r = window.document,
    t = window.localStorage,
    o = r.currentScript,
    s = o.getAttribute("data-api") || new URL(o.src).origin + "/api/event",
    l = t && t.plausible_ignore;
  function p(t) {
    console.warn("Ignoring Event: " + t);
  }
  function e(t, e) {
    if (
      /^localhost$|^127(\.[0-9]+){0,2}\.[0-9]+$|^\[::1?\]$/.test(a.hostname) ||
      "file:" === a.protocol
    )
      return p("localhost");
    if (
      !(
        window._phantom ||
        window.__nightmare ||
        window.navigator.webdriver ||
        window.Cypress
      )
    ) {
      if ("true" == l) return p("localStorage flag");
      var i = {};
      (i.n = t),
        (i.u = a.href),
        (i.d = o.getAttribute("data-domain")),
        (i.r = r.referrer || null),
        (i.w = window.innerWidth),
        e && e.meta && (i.m = JSON.stringify(e.meta)),
        e && e.props && (i.p = JSON.stringify(e.props));
      var n = new XMLHttpRequest();
      n.open("POST", s, !0),
        n.setRequestHeader("Content-Type", "text/plain"),
        n.send(JSON.stringify(i)),
        (n.onreadystatechange = function () {
          4 == n.readyState && e && e.callback && e.callback();
        });
    }
  }
  function i(t) {
    for (
      var e = t.target,
        i = "auxclick" == t.type && 2 == t.which,
        n = "click" == t.type;
      e && (void 0 === e.tagName || "a" != e.tagName.toLowerCase() || !e.href);
    )
      e = e.parentNode;
    e &&
      e.href &&
      e.host &&
      e.host !== a.host &&
      ((i || n) &&
        window.plausible("Outbound Link: Click", { props: { url: e.href } }),
      (e.target && !e.target.match(/^_(self|parent|top)$/i)) ||
        t.ctrlKey ||
        t.metaKey ||
        t.shiftKey ||
        !n ||
        (setTimeout(function () {
          a.href = e.href;
        }, 150),
        t.preventDefault()));
  }
  r.addEventListener("click", i), r.addEventListener("auxclick", i);
  var n = (window.plausible && window.plausible.q) || [];
  window.plausible = e;
  for (var c, d = 0; d < n.length; d++) e.apply(this, n[d]);
  function u() {
    c !== a.pathname && ((c = a.pathname), e("pageview"));
  }
  var w,
    h = window.history;
  h.pushState &&
    ((w = h.pushState),
    (h.pushState = function () {
      w.apply(this, arguments), u();
    }),
    window.addEventListener("popstate", u)),
    "prerender" === r.visibilityState
      ? r.addEventListener("visibilitychange", function () {
          c || "visible" !== r.visibilityState || u();
        })
      : u();
})();
/* eslint-enable */
