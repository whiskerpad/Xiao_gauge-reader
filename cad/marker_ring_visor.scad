// =====================================================================
// marker_ring_visor.scad  圧力計用マーカーリング+ひさし(gauge-reader)
// 単位: mm。OpenSCAD(無料)で開き、F6 → STLで書き出し。数値を変えれば寸法変更できる
//
// 座標: リングの正面がXY平面(+Yが12時方向)、+Z が手前(カメラ側)。計器は z<0 側
// 印刷: リングの裏面(z=0)を下にして、ひさしは上に伸びる向き(サポート不要)
// =====================================================================

// ---- 寸法 ----
ring_od      = 115;   // リング外径(計器の外形)
ring_id      = 95;    // リング内径(文字盤の直径)
ring_t       = 2.0;   // リングの厚さ

// ---- マーカー(黒丸を塗るためのくぼみ) ----
marker_r     = 52.5;  // マーカー中心の半径(ファームの mkr と同じ値にする)
marker_d     = 6.0;   // マーカーの直径
marker_depth = 0.6;   // くぼみの深さ(黒の油性ペンで塗る/フィラメント切替)
marker_ang   = [0, 90, 180, 270];  // 12時=0°、時計回り。ファームは現在この配置前提

// ---- ひさし(上側の庇) 0にすると無し ----
visor_depth  = 40;    // 手前への出っ張り
visor_span   = 120;   // 12時を中心とした角度範囲(度)
visor_t      = 2.0;   // ひさしの厚さ

$fn = 180;

module annulus(od, id, h) {
  difference() { cylinder(d = od, h = h); translate([0, 0, -1]) cylinder(d = id, h = h + 2); }
}

module sector(r, ang, h) {   // 12時を中心に ±ang/2 の扇形柱
  pts = concat([[0, 0]], [for (a = [-ang / 2 : 2 : ang / 2]) [r * sin(a), r * cos(a)]]);
  linear_extrude(height = h) polygon(pts);
}

difference() {
  union() {
    annulus(ring_od, ring_id, ring_t);
    if (visor_depth > 0)
      translate([0, 0, ring_t - 0.01])
        intersection() {
          annulus(ring_od, ring_od - 2 * visor_t, visor_depth);
          sector(ring_od, visor_span, visor_depth);
        }
  }
  for (a = marker_ang)
    translate([marker_r * sin(a), marker_r * cos(a), ring_t - marker_depth])
      cylinder(d = marker_d, h = marker_depth + 1);
}
