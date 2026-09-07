import pathlib

md = pathlib.Path("REPORT.md").read_text(encoding="utf-8")
safe = md.replace("</", "<\\/")

html = r'''<title>GAIA main Review</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{--bg:#F7F8F6;--surface:#FFFFFF;--ink:#1C2124;--ink-2:#4A5459;--rule:#D9DFDC;--accent:#0F6E6E;--accent-ink:#0B4F4F;--crit:#B3261E;--crit-bg:#FBEBEA;--imp:#9A6300;--imp-bg:#FBF3E1;--min:#2E7D32;--min-bg:#EAF4EB;--code-bg:#EEF2F0;--toc-bg:#F1F3F1}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15191B;--surface:#1D2225;--ink:#E6EAE8;--ink-2:#A9B3B0;--rule:#2E3639;--accent:#5FC1BE;--accent-ink:#8ED9D6;--crit:#F28B82;--crit-bg:#3A2321;--imp:#F0C15C;--imp-bg:#3A311C;--min:#8BD08F;--min-bg:#1F3322;--code-bg:#242B2E;--toc-bg:#191E20}}
:root[data-theme="dark"]{--bg:#15191B;--surface:#1D2225;--ink:#E6EAE8;--ink-2:#A9B3B0;--rule:#2E3639;--accent:#5FC1BE;--accent-ink:#8ED9D6;--crit:#F28B82;--crit-bg:#3A2321;--imp:#F0C15C;--imp-bg:#3A311C;--min:#8BD08F;--min-bg:#1F3322;--code-bg:#242B2E;--toc-bg:#191E20}
html{color-scheme:light dark}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Source Sans 3",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:17px;line-height:1.55}
.wrap{display:grid;grid-template-columns:260px minmax(0,1fr);gap:0;min-height:100vh}
nav.toc{position:sticky;top:0;align-self:start;height:100vh;overflow-y:auto;background:var(--toc-bg);border-right:1px solid var(--rule);padding:22px 18px;font-size:13.5px;box-sizing:border-box}
nav.toc .brand{font-family:Fraunces,Georgia,serif;font-weight:700;font-size:18px;letter-spacing:-.01em;margin:0 0 4px}
nav.toc .meta{color:var(--ink-2);font-family:"JetBrains Mono",ui-monospace,monospace;font-size:11.5px;margin:0 0 16px}
nav.toc a{display:block;color:var(--ink-2);text-decoration:none;padding:3px 0 3px 8px;border-left:2px solid transparent}
nav.toc a.l3{padding-left:20px;font-size:12.5px}
nav.toc a:hover,nav.toc a:focus{color:var(--accent-ink);border-left-color:var(--accent);outline:none}
main{padding:40px 56px 96px;max-width:960px;box-sizing:border-box}
article :is(h1,h2,h3){font-family:Fraunces,Georgia,serif;text-wrap:balance;letter-spacing:-.01em}
article h1{font-size:40px;font-weight:700;line-height:1.1;margin:0 0 12px}
article h2{font-size:27px;font-weight:700;margin:56px 0 14px;padding-top:18px;border-top:1px solid var(--rule)}
article h3{font-size:20px;font-weight:500;margin:34px 0 10px;color:var(--accent-ink)}
article p,article li{max-width:78ch}
article p{margin:0 0 14px}
article a{color:var(--accent-ink)}
article code{font-family:"JetBrains Mono",ui-monospace,Menlo,monospace;font-size:.86em;background:var(--code-bg);padding:1px 5px;border-radius:3px}
article pre{background:var(--code-bg);padding:14px 16px;border-radius:6px;overflow-x:auto;font-size:13px;line-height:1.45}
article pre code{background:none;padding:0}
article blockquote{margin:0 0 14px;padding:4px 16px;border-left:3px solid var(--accent);color:var(--ink-2)}
article ul,article ol{padding-left:22px}
article li{margin:0 0 8px}
article hr{border:0;border-top:1px solid var(--rule);margin:36px 0}
.tablewrap{overflow-x:auto;margin:0 0 18px}
article table{border-collapse:collapse;font-size:14px;min-width:520px}
article th,article td{border:1px solid var(--rule);padding:7px 10px;vertical-align:top;text-align:left}
article th{background:var(--toc-bg);font-weight:600}
article td{font-variant-numeric:tabular-nums}
.f{padding:14px 16px 12px 18px;margin:0 0 16px;border-left:4px solid var(--rule);background:var(--surface);border-radius:0 6px 6px 0;list-style:none}
.f.crit{border-left-color:var(--crit)}
.f.imp{border-left-color:var(--imp)}
.f.crit>strong{color:var(--crit)}
.f.imp>strong{color:var(--imp)}
.pill{display:inline-block;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;padding:2px 7px;border-radius:3px;margin-right:8px;vertical-align:2px}
.pill.crit{background:var(--crit-bg);color:var(--crit)}.pill.imp{background:var(--imp-bg);color:var(--imp)}.pill.min{background:var(--min-bg);color:var(--min)}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:0 0 34px}
.stat{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:12px 14px}
.stat .n{font-family:Fraunces,Georgia,serif;font-size:30px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}
.stat .l{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-2);margin-top:6px}
.stat.crit .n{color:var(--crit)}.stat.imp .n{color:var(--imp)}.stat.min .n{color:var(--min)}
@media (max-width:900px){.wrap{grid-template-columns:1fr}nav.toc{position:static;height:auto;border-right:0;border-bottom:1px solid var(--rule)}main{padding:24px 18px 64px}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (prefers-reduced-motion:no-preference){nav.toc a{transition:color .15s,border-color .15s}}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<div class="wrap">
<nav class="toc" aria-label="Contents"><p class="brand">GAIA main Review</p><p class="meta">amd/gaia @ 211f08c5 · v0.23.1 · 2026-09-04</p><div id="toc"></div></nav>
<main><div class="stats" id="stats"></div><article id="a"></article></main>
</div>
<script type="text/markdown" id="src">__MD__</script>
<script>
(function(){
  var src=document.getElementById('src').textContent;
  marked.setOptions({gfm:true,breaks:false});
  var html=marked.parse(src);
  var a=document.getElementById('a'); a.innerHTML=html;
  a.querySelectorAll('table').forEach(function(t){var w=document.createElement('div');w.className='tablewrap';t.parentNode.insertBefore(w,t);w.appendChild(t);});
  a.querySelectorAll('p,li').forEach(function(el){
    var s=el.firstElementChild; if(!s||s.tagName!=='STRONG'||el.firstChild!==s) return;
    var m=/^([CI])(\d+)\./.exec(s.textContent); if(!m) return;
    var cls=m[1]==='C'?'crit':'imp'; el.classList.add('f',cls);
    var pill=document.createElement('span'); pill.className='pill '+cls; pill.textContent=(m[1]==='C'?'critical ':'important ')+m[1]+m[2];
    el.insertBefore(pill,s); s.textContent=s.textContent.replace(/^[CI]\d+\.\s*/,'');
    el.id=m[1]+m[2];
  });
  var toc=document.getElementById('toc'); var used={};
  a.querySelectorAll('h2,h3').forEach(function(h){
    var id=h.textContent.toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');
    if(used[id]){id+='-'+(++used[id]);}else{used[id]=1;}
    h.id=id; var l=document.createElement('a'); l.href='#'+id; l.textContent=h.textContent.replace(/^Appendix /,'App. ');
    if(h.tagName==='H3') l.className='l3'; toc.appendChild(l);
  });
  var counts={crit:(src.match(/^\*\*C\d+\./gm)||[]).length, imp:(src.match(/^- \*\*I\d+\./gm)||[]).length};
  var st=document.getElementById('stats');
  [['crit',counts.crit,'critical findings'],['imp',counts.imp,'important findings'],['min','≈120','minor items'],['','37','probe-reproduced']].forEach(function(x){
    var d=document.createElement('div'); d.className='stat '+x[0]; d.innerHTML='<div class="n">'+x[1]+'</div><div class="l">'+x[2]+'</div>'; st.appendChild(d);
  });
  var walker=document.createTreeWalker(a,NodeFilter.SHOW_TEXT,null); var nodes=[];
  while(walker.nextNode()){var n=walker.currentNode; if(n.parentNode.closest('code,pre,a,h1,h2,h3,.pill')) continue; if(/\b[CI]\d{1,2}\b/.test(n.nodeValue)) nodes.push(n);}
  nodes.forEach(function(n){var frag=document.createDocumentFragment(); var re=/\b([CI])(\d{1,2})\b/g; var last=0,m,s=n.nodeValue;
    while((m=re.exec(s))){ if(!document.getElementById(m[1]+m[2])) continue; frag.appendChild(document.createTextNode(s.slice(last,m.index))); var l=document.createElement('a'); l.href='#'+m[1]+m[2]; l.textContent=m[0]; frag.appendChild(l); last=m.index+m[0].length; }
    if(last===0) return; frag.appendChild(document.createTextNode(s.slice(last))); n.parentNode.replaceChild(frag,n);});
})();
</script>'''

out = html.replace("__MD__", safe)
pathlib.Path("artifact.html").write_text(out, encoding="utf-8")
print("ok", len(out))
