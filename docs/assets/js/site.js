/* ============================================================
   站点共享脚本
   - 复制按钮 / 图片灯箱 / 滚动渐显 / 数字动画
   - 导航与章节锚点高亮
   - 门户项目卡片渲染（读 assets/data/projects.js）
   所有功能都做了缺元素保护，同一份脚本可被任意页面安全加载。
   ============================================================ */
(function () {
  "use strict";

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  /* ---------- toast ---------- */
  var toast = $("#toast"), tTimer;
  function showToast(msg) {
    if (!toast) return;
    toast.textContent = msg || "已复制";
    toast.classList.add("on");
    clearTimeout(tTimer);
    tTimer = setTimeout(function () { toast.classList.remove("on"); }, 1700);
  }

  /* ---------- 复制 ---------- */
  function copyText(text, cb) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(
        function () { cb(true); },
        function () { fallback(text, cb); }
      );
    } else { fallback(text, cb); }
  }
  function fallback(text, cb) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); cb(true); } catch (e) { cb(false); }
    document.body.removeChild(ta);
  }
  $$("[data-copy],[data-copy-text]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = btn.getAttribute("data-copy-text");
      if (!text) {
        var el = $(btn.getAttribute("data-copy"));
        text = el ? el.innerText : "";
      }
      copyText(text, function (ok) {
        var old = btn.textContent;
        btn.textContent = ok ? "已复制" : "复制失败";
        showToast(ok ? "已复制到剪贴板" : "复制失败，请手动选择");
        setTimeout(function () { btn.textContent = old; }, 1600);
      });
    });
  });

  /* ---------- 图片灯箱 ---------- */
  var lb = $("#lb"), lbimg = $("#lbimg");
  if (lb && lbimg) {
    $$("img[data-lb]").forEach(function (img) {
      img.addEventListener("click", function () {
        lbimg.src = img.src;
        lbimg.alt = img.alt || "";
        lb.classList.add("on");
        document.body.style.overflow = "hidden";
      });
    });
    var closeLb = function () {
      lb.classList.remove("on");
      lbimg.src = "";
      document.body.style.overflow = "";
    };
    lb.addEventListener("click", closeLb);
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && lb.classList.contains("on")) closeLb();
    });
  }

  /* ---------- 滚动渐显 ---------- */
  var rv = $$(".rv");
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
    rv.forEach(function (el) { io.observe(el); });
  } else {
    rv.forEach(function (el) { el.classList.add("in"); });
  }

  /* ---------- 章节锚点高亮（页内 section 链接） ---------- */
  var secs = $$("section[id]");
  var secLinks = $$(".nav-links a[href^='#'], .anchorbar a[href^='#']");
  function syncAnchor() {
    if (!secs.length) return;
    var y = window.scrollY + 130, cur = "";
    secs.forEach(function (s) {
      if (s.offsetTop <= y) cur = s.id;
    });
    secLinks.forEach(function (a) {
      a.classList.toggle("on", a.getAttribute("href") === "#" + cur);
      a.classList.toggle("active", a.getAttribute("href") === "#" + cur);
    });
  }
  window.addEventListener("scroll", syncAnchor, { passive: true });
  syncAnchor();

  /* ---------- 当前页导航高亮（跨页链接） ---------- */
  (function () {
    var path = location.pathname.replace(/\/index\.html$/, "/");
    $$(".nav-links a").forEach(function (a) {
      var href = a.getAttribute("href") || "";
      if (href.charAt(0) === "#" || href.indexOf("http") === 0) return;
      var abs = new URL(href, location.href).pathname.replace(/\/index\.html$/, "/");
      if (abs === path) a.classList.add("active");
    });
  })();

  /* ---------- 数字滚动 ---------- */
  function animateCount(el) {
    var target = parseFloat(el.getAttribute("data-count"));
    if (isNaN(target)) return;
    var dec = parseInt(el.getAttribute("data-dec") || "0", 10);
    var suffix = el.getAttribute("data-suffix") || "";
    var dur = 900, t0 = null;
    function frame(ts) {
      if (!t0) t0 = ts;
      var p = Math.min((ts - t0) / dur, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      var v = target * eased;
      el.textContent = (dec > 0 ? v.toFixed(dec) : Math.round(v).toLocaleString()) + suffix;
      if (p < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  var statEls = $$(".stat .n[data-count]");
  if ("IntersectionObserver" in window && statEls.length) {
    var io2 = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { animateCount(en.target); io2.unobserve(en.target); }
      });
    }, { threshold: 0.4 });
    statEls.forEach(function (el) { io2.observe(el); });
  } else {
    statEls.forEach(animateCount);
  }

  /* ---------- 门户：档案渲染 + 方向筛选 + 搜索 ---------- */
  var grid = $("#pgrid");
  if (grid && window.REPRODUCTIONS) {
    var esc = function (s) {
      return String(s).replace(/[&<>"]/g, function (c) {
        return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
      });
    };
    var list = window.REPRODUCTIONS;
    var label = {}, order = [];
    (window.CATEGORIES || []).forEach(function (c) { label[c.id] = c.label; order.push(c.id); });
    var nameOf = function (id) { return label[id] || id; };

    /* 用到的方向：按登记表顺序排；未登记的方向排在后面但照常显示 */
    var used = [], counts = {};
    list.forEach(function (p) {
      (p.cats || []).forEach(function (c) {
        if (used.indexOf(c) < 0) { used.push(c); counts[c] = 0; }
        counts[c]++;
      });
    });
    used.sort(function (a, b) {
      var ia = order.indexOf(a), ib = order.indexOf(b);
      return (ia < 0 ? 999 : ia) - (ib < 0 ? 999 : ib);
    });

    var catsBox = $("#cats"), countEl = $("#count"), emptyEl = $("#empty"),
        resetEl = $("#reset"), qEl = $("#q"), metaline = $("#metaline");
    var state = { cat: "all", q: "" };

    if (metaline) {
      metaline.textContent = list.length + " 篇 · " + used.length + " 个方向";
    }

    /* 筛选条 */
    if (catsBox) {
      catsBox.innerHTML =
        '<button class="catbtn on" type="button" data-cat="all" aria-pressed="true">全部<sup>' +
          list.length + "</sup></button>" +
        used.map(function (c) {
          return '<button class="catbtn" type="button" data-cat="' + esc(c) + '" aria-pressed="false">' +
            esc(nameOf(c)) + "<sup>" + counts[c] + "</sup></button>";
        }).join("");
      catsBox.addEventListener("click", function (e) {
        var btn = e.target.closest ? e.target.closest(".catbtn") : null;
        if (!btn) return;
        state.cat = btn.getAttribute("data-cat");
        $$(".catbtn", catsBox).forEach(function (b) {
          var on = b === btn;
          b.classList.toggle("on", on);
          b.setAttribute("aria-pressed", on ? "true" : "false");
        });
        draw();
      });
    }

    /* 搜索：标题、摘要、期刊、方向名、技术关键词全进索引 */
    function haystack(p) {
      var parts = [p.title, p.sum, p.venue, p.statusText]
        .concat((p.cats || []).map(nameOf))
        .concat(p.cats || [])
        .concat(p.tags || []);
      return parts.filter(Boolean).join(" ").toLowerCase();
    }
    function matched(p) {
      if (state.cat !== "all" && (p.cats || []).indexOf(state.cat) < 0) return false;
      if (!state.q) return true;
      return haystack(p).indexOf(state.q) >= 0;
    }

    function card(p, i) {
      var no = String(p.no || (i + 1)).padStart(2, "0");
      var chips = (p.cats || []).map(function (c) {
        return "<span>" + esc(nameOf(c)) + "</span>";
      }).join("");
      return (
        '<a class="pcard" href="' + esc(p.href) + '">' +
          '<div class="cover">' +
            '<img src="' + esc(p.cover) + '" alt="' + esc(p.title) + '" loading="lazy">' +
            '<span class="pno">' + no + "</span>" +
          "</div>" +
          '<div class="body">' +
            '<div class="topline">' +
              '<span class="venue">' + esc(p.venue) + "</span>" +
              '<span class="badge ' + esc(p.status) + '">' + esc(p.statusText) + "</span>" +
            "</div>" +
            "<h3>" + esc(p.title) + "</h3>" +
            '<p class="sum">' + esc(p.sum) + "</p>" +
            '<div class="cats">' + chips + "</div>" +
            '<span class="go">阅读复现记录 <span aria-hidden="true">&rarr;</span></span>' +
          "</div>" +
        "</a>"
      );
    }

    function draw() {
      var hits = list.filter(matched);
      grid.innerHTML = hits.map(card).join("");
      if (emptyEl) emptyEl.hidden = hits.length > 0;
      if (countEl) {
        var filtering = state.cat !== "all" || state.q;
        countEl.textContent = filtering
          ? "筛出 " + hits.length + " 篇 · 共 " + list.length + " 篇"
          : "共 " + list.length + " 篇";
      }
      if (resetEl) resetEl.hidden = !(state.cat !== "all" || state.q);
    }

    if (qEl) {
      qEl.addEventListener("input", function () {
        state.q = qEl.value.trim().toLowerCase();
        draw();
      });
      /* 按 / 聚焦搜索，Esc 清空 */
      document.addEventListener("keydown", function (e) {
        if (e.key === "/" && document.activeElement !== qEl) {
          e.preventDefault();
          qEl.focus();
        } else if (e.key === "Escape" && document.activeElement === qEl) {
          qEl.value = ""; state.q = ""; draw(); qEl.blur();
        }
      });
    }
    if (resetEl) {
      resetEl.addEventListener("click", function () {
        state = { cat: "all", q: "" };
        if (qEl) qEl.value = "";
        $$(".catbtn", catsBox).forEach(function (b, i) {
          b.classList.toggle("on", i === 0);
          b.setAttribute("aria-pressed", i === 0 ? "true" : "false");
        });
        draw();
      });
    }

    draw();
  }
})();
