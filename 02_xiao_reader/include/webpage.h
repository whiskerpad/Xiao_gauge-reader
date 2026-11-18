#pragma once
// 設定・確認用Web画面
//  - ボタンで項目を選び画像をクリックして設定 → 保存
//  - 矢印キー: 中心を1px移動(Shift併用で5px)。内半径/外半径選択中は↑↓で半径を変更
//  - マーカー: 有効にし、M上/M右/M下/M左 を選んで画像上の黒丸をクリック → 設定保存 → マーカー検出
//              検出OKの後、中心(針の軸)・0目盛り・最大目盛りをクリック(文字盤平面[mm]に変換して保存)
//  - 背景: リセット → 針を動かしながら「背景追加」を繰り返す → 背景保存
//          (PCの record_background.py で自動化可能)
static const char INDEX_HTML[] = R"rawliteral(<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Gauge Reader</title>
<style>body{font-family:sans-serif;margin:8px;background:#222;color:#eee}
canvas{max-width:100%;border:1px solid #666}#cv{cursor:crosshair}button{margin:2px;padding:6px 10px}
.on{background:#4a8;color:#fff}input{width:4.5em}#nums input{background:#333;color:#fff;border:1px solid #777}.row{margin:4px 0}
#info{font-family:monospace;white-space:pre-wrap;font-size:12px}.warn{color:#fc6}</style></head><body>
<div class="row">
<button data-m="c">中心</button><button data-m="ri">内半径</button><button data-m="ro">外半径</button>
<button data-m="z">0目盛り</button><button data-m="m">最大目盛り</button>
最小値<input id="vmin"> 最大値<input id="vmax">
<button id="save">設定保存</button><button id="shot">再撮影</button><button id="read">読み取り</button>
</div>
<div class="row" id="nums">
中心X<input id="n_cx"> 中心Y<input id="n_cy"> 内半径<input id="n_ri"> 外半径<input id="n_ro">
0目盛り角<input id="n_a0"> 最大目盛り角<input id="n_a1"> 振れ角<span id="n_span"></span>°
 直線補正 <span id="n_corr"></span>
</div>
<div class="row" id="mb">多帯(背景なし):
<label><input type="checkbox" id="c_nb" style="width:auto">有効</label>
帯1<input id="n_b1i">〜<input id="n_b1o"> 帯2<input id="n_b2i">〜<input id="n_b2o"> 帯3<input id="n_b3i">〜<input id="n_b3o">
ハイパス±<input id="n_hpw">° <span class="warn">(有効時は内半径/外半径・背景は使わない)</span></div>
<div class="row" id="mkrow">マーカー:
<label><input type="checkbox" id="c_mk" style="width:auto">有効</label>
<label><input type="checkbox" id="c_mkp" style="width:auto">白マーカー(黒地)</label>
<button data-m="k0">M上</button><button data-m="k1">M右</button><button data-m="k2">M下</button><button data-m="k3">M左</button>
<button id="mkdet">マーカー検出</button><button id="mkfind">自動探索</button> マーカー半径<input id="n_mkr">mm 探索窓<input id="n_mkw">px
<span id="mkst"></span><br>
帯[mm] 1:<input id="n_m1i">〜<input id="n_m1o"> 2:<input id="n_m2i">〜<input id="n_m2o"> 3:<input id="n_m3i">〜<input id="n_m3o">
 回転中心[mm]<input id="n_pcx">,<input id="n_pcy"> 0目盛り角<input id="n_ma0"> 最大目盛り角<input id="n_ma1">
 <span class="warn">(有効時は中心・0目盛り・最大目盛りのクリックが文字盤平面の値になる)</span><br>
<label><input type="checkbox" id="c_rgon" style="width:auto">リング方式</label> 半径[mm]<input id="n_rg0">〜<input id="n_rg1"> 割合<input id="n_rgq">
 <span class="warn">(有効時は帯1〜3は使わない。1mm幅のリングの7割以上で暗い角度を針とする=割合0.3)</span><br>
<label><input type="checkbox" id="c_fpon" style="width:auto">置針を除外</label> 置針の位置(目盛り値)<input id="n_fpv"> 幅±<input id="n_fpw">°
 <span id="fpst"></span><br>
視差補正: 針の奥行き<input id="n_nd">mm(マーカー面から針まで、0=補正なし) 焦点距離<input id="n_camf">px
 <span class="warn">(斜めから撮るとき用。回転中心は補正後の値で保存される)</span></div>
<div class="row">露出:
<label><input type="checkbox" id="c_auto" style="width:auto">自動</label>
露光量<input id="c_aec">(0-<span id="c_aecmax"></span>) ゲイン<input id="c_gain">(0-<span id="c_gainmax"></span>)
<button id="camset">適用</button> <span id="camst"></span></div>
<div class="row">背景:
<button id="bgreset">リセット</button> 針位置<input id="bgskip" placeholder="MPa"><button id="bgadd">追加</button><button id="bgsave">保存</button>
<button id="bgclear">削除</button> <span id="bgst"></span></div>
<div class="row warn" id="dirty"></div>
<canvas id="cv"></canvas>
<canvas id="pf" width="800" height="200"></canvas>
<div id="info"></div>
<script>
let cfg={},mode='c',res=null,dirty=false,mkH=null;const img=new Image(),cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const $=id=>document.getElementById(id);
const J=async u=>(await fetch(u)).json();
document.querySelectorAll('[data-m]').forEach(b=>b.onclick=()=>{mode=b.dataset.m;mark();});
function mark(){document.querySelectorAll('[data-m]').forEach(b=>b.classList.toggle('on',b.dataset.m==mode));}
function setDirty(v){dirty=v;$('dirty').textContent=v?'未保存の変更あり(設定保存で確定。中心・半径・目盛りを変えると背景は削除されます)':'';}
function ang(x,y){return (Math.atan2(x-cfg.cx,-(y-cfg.cy))*180/Math.PI+360)%360;}
function dist(x,y){return Math.hypot(x-cfg.cx,y-cfg.cy);}
function r1(v){return Math.round(v*10)/10;}
function load(src){img.onload=()=>{cv.width=img.width;cv.height=img.height;draw();};img.src=src+'?t='+Date.now();}
function ray(a,r0,r1_,col){const t=a*Math.PI/180;ctx.strokeStyle=col;ctx.beginPath();
 ctx.moveTo(cfg.cx+r0*Math.sin(t),cfg.cy-r0*Math.cos(t));ctx.lineTo(cfg.cx+r1_*Math.sin(t),cfg.cy-r1_*Math.cos(t));ctx.stroke();}
const KEYS=['cx','cy','ri','ro','a0','a1','b1i','b1o','b2i','b2o','b3i','b3o','hpw'];
const MKEYS=['mkr','mkw','m1i','m1o','m2i','m2o','m3i','m3o','pcx','pcy','ma0','ma1','rg0','rg1','rgq','fpv','fpw','nd','camf'];
function r2(v){return Math.round(v*100)/100;}
function hmap(H,x,y){const w=H[6]*x+H[7]*y+H[8];return [(H[0]*x+H[1]*y+H[2])/w,(H[3]*x+H[4]*y+H[5])/w];}
async function mkStatus(det){mkH=await J(det?'/mk/detect':'/mk');
 const FN=['上','右','下','左'],FR=['','窓が画像の外','コントラスト不足(黒丸が見えない)','大きさが黒丸と違う(影・縁とつながる/位置ずれ)','変換失敗(4点の配置が不正)','縮尺異常','形が不正(追跡が別の点に移った→自動探索)','紙リングが見えない(追跡が別の点に移った→自動探索)'];
 const why=mkH.ok?'':('('+(mkH.fail_idx>=0?FN[mkH.fail_idx]+': ':'')+(FR[mkH.fail_code]||'')+')');
 $('mkst').textContent=(mkH.ok?'検出OK':'未検出'+why)+'  縮尺 '+mkH.ppm.toFixed(2)+'px/mm  コントラスト '+mkH.contrast.map(v=>v.toFixed(2)).join(',');
 if(det)load('/last.jpg');else draw();}
function showNums(){KEYS.forEach(k=>{const e=$('n_'+k);if(document.activeElement!==e)e.value=cfg[k];});
 if(document.activeElement!==$('c_nb'))$('c_nb').checked=(cfg.nb|0)>=2;
 MKEYS.forEach(k=>{const e=$('n_'+k);if(document.activeElement!==e)e.value=cfg[k];});
 $('c_mk').checked=!!cfg.mk;$('c_mkp').checked=!!cfg.mkp;
 $('c_rgon').checked=!!cfg.rgon;$('c_fpon').checked=!!cfg.fpon;
 $('fpst').textContent=cfg.fpon?('置針の暗さ(学習値) '+((cfg.fpl??-1)<0?'未学習':cfg.fpl.toFixed(1)+'%')):'';
 $('n_corr').textContent='a='+(cfg.ca??1)+' b='+(cfg.cb??0);
 const A0=cfg.mk?cfg.ma0:cfg.a0,A1=cfg.mk?cfg.ma1:cfg.a1;
 $('n_span').textContent=r1(((A1-A0)%360+360)%360||360);}
$('c_mk').onchange=e=>{cfg.mk=e.target.checked?1:0;setDirty(true);draw();};
$('c_mkp').onchange=e=>{cfg.mkp=e.target.checked?1:0;setDirty(true);draw();};
$('c_rgon').onchange=e=>{cfg.rgon=e.target.checked?1:0;setDirty(true);draw();};
$('c_fpon').onchange=e=>{cfg.fpon=e.target.checked?1:0;setDirty(true);draw();};
MKEYS.forEach(k=>{$('n_'+k).onchange=e=>{const v=parseFloat(e.target.value);if(!isNaN(v)){cfg[k]=(k=='mkw')?Math.round(v):r2(v);setDirty(true);draw();}};});
$('mkdet').onclick=()=>{if(dirty){alert('先に設定保存してください');return;}mkStatus(true);};
$('mkfind').onclick=async()=>{if(dirty){alert('先に設定保存してください');return;}$('mkst').textContent='自動探索中…';
 const r=await J('/mk/find');cfg=await J('/config');await mkStatus(false);
 $('mkst').textContent=(r.find_ok?'自動探索OK  ':'自動探索失敗  ')+r.find_info+'  / '+$('mkst').textContent;load('/last.jpg');};
$('c_nb').onchange=e=>{cfg.nb=e.target.checked?3:0;setDirty(true);draw();};
KEYS.forEach(k=>{$('n_'+k).onchange=e=>{const v=parseFloat(e.target.value);if(!isNaN(v)){cfg[k]=r1(v);setDirty(true);draw();}};});
function drawMk(){
 ctx.strokeStyle='#888';for(let i=0;i<4;i++)ctx.strokeRect(cfg['mu'+i]-cfg.mkw,cfg['mv'+i]-cfg.mkw,2*cfg.mkw,2*cfg.mkw);
 if(mkH){ctx.strokeStyle=mkH.ok?'#0f0':'#f00';ctx.fillStyle=ctx.strokeStyle;ctx.font='14px sans-serif';
  for(let i=0;i<4;i++){const u=mkH.u[i],v=mkH.v[i];ctx.beginPath();ctx.moveTo(u-8,v);ctx.lineTo(u+8,v);ctx.moveTo(u,v-8);ctx.lineTo(u,v+8);ctx.stroke();
   ctx.fillText('上右下左'[i],u+6,v-6);}}
 if(!(mkH&&mkH.ok))return;
 const H=mkH.H,qx=cfg.pcx+(mkH.sdx||0),qy=cfg.pcy+(mkH.sdy||0),pc=(r,a)=>{const t=a*Math.PI/180;return hmap(H,qx+r*Math.sin(t),qy-r*Math.cos(t));};
 const circ=(r,col)=>{ctx.strokeStyle=col;ctx.beginPath();for(let a=0;a<=360;a+=5){const q=pc(r,a);a?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);}ctx.stroke();};
 const rayP=(a,ra,rb,col)=>{const p0=pc(ra,a),p1=pc(rb,a);ctx.strokeStyle=col;ctx.beginPath();ctx.moveTo(p0[0],p0[1]);ctx.lineTo(p1[0],p1[1]);ctx.stroke();};
 const C=['#f0f','#0ff','#ff0'],RO=cfg.rgon?cfg.rg1:cfg.m3o,RI=cfg.rgon?cfg.rg0:cfg.m1i;
 if(cfg.rgon){circ(cfg.rg0,'#0ff');circ(cfg.rg1,'#0ff');}
 else [[cfg.m1i,cfg.m1o],[cfg.m2i,cfg.m2o],[cfg.m3i,cfg.m3o]].forEach((b,i)=>b.forEach(r=>circ(r,C[i])));
 rayP(cfg.ma0,0,RO,'#09f');rayP(cfg.ma1,0,RO,'#f90');
 if(cfg.fpon){const sp=((cfg.ma1-cfg.ma0)%360+360)%360||360,fa=cfg.ma0+(cfg.fpv-cfg.vmin)/(cfg.vmax-cfg.vmin)*sp;
  ctx.setLineDash([4,4]);rayP(fa-cfg.fpw,RI,RO*1.15,'#f80');rayP(fa+cfg.fpw,RI,RO*1.15,'#f80');ctx.setLineDash([]);}
 const c0=pc(0,0);ctx.fillStyle='#0f0';ctx.fillRect(c0[0]-1,c0[1]-1,3,3);
 if(res&&res.ok)rayP(res.angle,RI,RO*1.1,'#f00');}
function draw(){showNums();if(!img.width)return;ctx.drawImage(img,0,0);ctx.lineWidth=1.5;ctx.strokeStyle='#0f0';
 if(cfg.mk){drawMk();$('info').textContent='config '+JSON.stringify(cfg)+'\nresult '+(res?JSON.stringify(res):'-');return;}
 if((cfg.nb|0)>=2){const C=['#f0f','#0ff','#ff0'];
  [[cfg.b1i,cfg.b1o],[cfg.b2i,cfg.b2o],[cfg.b3i,cfg.b3o]].forEach((b,i)=>{ctx.strokeStyle=C[i];
   b.forEach(r=>{ctx.beginPath();ctx.arc(cfg.cx,cfg.cy,r,0,2*Math.PI);ctx.stroke();});});
  ctx.strokeStyle='#0f0';}
 else [cfg.ri,cfg.ro].forEach(r=>{ctx.beginPath();ctx.arc(cfg.cx,cfg.cy,r,0,2*Math.PI);ctx.stroke();});
 ctx.fillStyle='#0f0';ctx.fillRect(cfg.cx-1,cfg.cy-1,3,3);
 ray(cfg.a0,0,cfg.ro,'#09f');ray(cfg.a1,0,cfg.ro,'#f90');
 if(res&&res.ok)ray(res.angle,cfg.ri,cfg.ro*1.1,'#f00');
 $('info').textContent='config '+JSON.stringify(cfg)+'\nresult '+(res?JSON.stringify(res):'-');}
async function bgStatus(){const b=await J('/bg');$('bgst').textContent=(b.valid?'有効':'未使用')+'  取込フレーム数='+b.count;}
async function plot(){const p=await J('/profile'),c=$('pf'),g=c.getContext('2d'),W=c.width,H=c.height;
 g.fillStyle='#111';g.fillRect(0,0,W,H);const n=p.values.length;if(!n)return;
 const X=i=>i/(n-1)*W,Y=v=>H-v/255*H;
 const line=(a,col,f)=>{if(!a)return;g.strokeStyle=col;g.beginPath();a.forEach((v,i)=>{const y=f(v);i?g.lineTo(X(i),y):g.moveTo(X(i),y);});g.stroke();};
 line(p.bg,'#0c0',Y);line(p.values,'#ccc',Y);
 const dm=Math.max(1,...p.dark);line(p.dark,'#f44',v=>H-Math.max(0,v)/dm*H*0.9);
 const xr=(p.rel-p.start)/p.step;g.strokeStyle='#ff0';g.beginPath();g.moveTo(X(xr),0);g.lineTo(X(xr),H);g.stroke();
 g.fillStyle='#ccc';g.font='12px sans-serif';g.fillText('灰:輝度 緑:背景 赤:暗さ(正規化) 黄:検出角  多帯='+p.multi+'  背景使用='+p.bg_used+'  gain='+p.gain+'  白飛び='+Math.round(p.sat*100)+'%  輝度最大='+Math.round(Math.max(...p.values)),4,14);}
cv.onclick=e=>{const b=cv.getBoundingClientRect(),x=(e.clientX-b.left)*cv.width/b.width,y=(e.clientY-b.top)*cv.height/b.height;
 if(mode[0]=='k'){const i=+mode[1];cfg['mu'+i]=r1(x);cfg['mv'+i]=r1(y);setDirty(true);draw();return;}
 if(cfg.mk){if(!(mkH&&mkH.ok)){alert('先にマーカー検出(検出OK)してください');return;}
  const P=hmap(mkH.Hinv,x,y);
  if(!isFinite(P[0])||!isFinite(P[1])||Math.hypot(P[0],P[1])>cfg.mkr){alert('変換結果が文字盤の外になりました。マーカー検出をやり直してください');return;}
  if(mode=='c'){cfg.pcx=r2(P[0]-(mkH.sdx||0));cfg.pcy=r2(P[1]-(mkH.sdy||0));}
  else if(mode=='z'||mode=='m'){const ax=cfg.pcx+(mkH.sdx||0),ay=cfg.pcy+(mkH.sdy||0),a=r1((Math.atan2(P[0]-ax,-(P[1]-ay))*180/Math.PI+360)%360);if(mode=='z')cfg.ma0=a;else cfg.ma1=a;}
  else return;
  setDirty(true);draw();return;}
 if(mode=='c'){cfg.cx=x;cfg.cy=y;}else if(mode=='ri')cfg.ri=dist(x,y);else if(mode=='ro')cfg.ro=dist(x,y);
 else if(mode=='z')cfg.a0=ang(x,y);else if(mode=='m')cfg.a1=ang(x,y);
 for(const k of['cx','cy','ri','ro','a0','a1'])cfg[k]=r1(cfg[k]);setDirty(true);draw();};
document.addEventListener('keydown',e=>{if(e.target.tagName=='INPUT')return;
 if(cfg.mk&&mode=='c'){const d=e.shiftKey?0.5:0.1;let u=true;
  if(e.key=='ArrowLeft')cfg.pcx-=d;else if(e.key=='ArrowRight')cfg.pcx+=d;else if(e.key=='ArrowUp')cfg.pcy-=d;else if(e.key=='ArrowDown')cfg.pcy+=d;else u=false;
  if(u){e.preventDefault();cfg.pcx=r2(cfg.pcx);cfg.pcy=r2(cfg.pcy);setDirty(true);draw();}return;}
 const d=e.shiftKey?5:1;let used=true;
 if(mode=='ri'||mode=='ro'){if(e.key=='ArrowUp')cfg[mode]+=d;else if(e.key=='ArrowDown')cfg[mode]-=d;else used=false;}
 else if(e.key=='ArrowLeft')cfg.cx-=d;else if(e.key=='ArrowRight')cfg.cx+=d;
 else if(e.key=='ArrowUp')cfg.cy-=d;else if(e.key=='ArrowDown')cfg.cy+=d;else used=false;
 if(used){e.preventDefault();for(const k of['cx','cy','ri','ro'])cfg[k]=r1(cfg[k]);setDirty(true);draw();}});
$('save').onclick=async()=>{cfg.vmin=+$('vmin').value;cfg.vmax=+$('vmax').value;
 cfg=await J('/config/set?'+new URLSearchParams(cfg));setDirty(false);draw();bgStatus();};
$('shot').onclick=()=>{res=null;load('/capture.jpg');};
$('read').onclick=async()=>{if(dirty){alert('先に設定保存してください');return;}res=await J('/read?n=3');
 if(cfg.mk)mkH=await J('/mk');load('/last.jpg');plot();};
$('bgreset').onclick=async()=>{await J('/bg/reset');bgStatus();};
$('bgadd').onclick=async()=>{if(dirty){alert('先に設定保存してください');return;}
 const v=$('bgskip').value.trim();if(v===''){alert('針位置(現在の針の値 MPa)を入力してください。その前後は背景から除外されます');return;}
 await J('/bg/add?n=2&skipw=15&skipv='+encodeURIComponent(v));bgStatus();load('/last.jpg');};
async function camStatus(){const c=await J('/cam');$('c_auto').checked=!!c.auto;$('c_aec').value=c.aec;$('c_gain').value=c.gain;
 $('c_aecmax').textContent=c.aec_max;$('c_gainmax').textContent=c.gain_max;
 $('camst').textContent=c.sensor+'  '+(c.auto?'自動露出':'固定');}
$('camset').onclick=async()=>{if(!confirm('露出を変更すると背景は削除されます。よろしいですか'))return;
 await J('/cam/set?auto='+($('c_auto').checked?1:0)+'&aec='+(+$('c_aec').value)+'&gain='+(+$('c_gain').value));
 camStatus();bgStatus();setTimeout(()=>load('/capture.jpg'),600);};
$('bgsave').onclick=async()=>{const r=await fetch('/bg/save');if(!r.ok)alert('保存失敗: '+(await r.text()));bgStatus();};
$('bgclear').onclick=async()=>{await J('/bg/clear');bgStatus();};
(async()=>{cfg=await J('/config');$('vmin').value=cfg.vmin;$('vmax').value=cfg.vmax;mark();load('/capture.jpg');bgStatus();camStatus();mkStatus(false);})();
</script></body></html>)rawliteral";
