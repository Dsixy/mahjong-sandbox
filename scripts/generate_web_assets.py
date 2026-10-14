"""Original SVG Mahjong faces and portraits. No downloaded artwork or fonts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "mahjong_sandbox/web/assets"
GREEN, RED, BLUE = "#17654c", "#ab3038", "#213f75"
# Common Chinese tile faces. Keep the two-and-four groups visually distinct;
# colour identifies groups, not individual pips within the same group.
DOT_POSITIONS = {
    1: [(32, 44)], 2: [(32, 27), (32, 61)], 3: [(19, 22), (32, 44), (45, 66)],
    4: [(19, 27), (45, 27), (19, 61), (45, 61)],
    5: [(18, 23), (46, 23), (32, 44), (18, 65), (46, 65)],
    6: [(20, 20), (44, 20), (20, 51), (44, 51), (20, 70), (44, 70)],
    7: [(18, 16), (32, 27), (46, 38), (20, 57), (44, 57), (20, 73), (44, 73)],
    8: [(x, y) for y in (18, 35, 53, 70) for x in (20, 44)],
    9: [(x, y) for y in (21, 44, 67) for x in (16, 32, 48)],
}
DOT_COLORS = {
    1: (GREEN,), 2: (GREEN,) * 2, 3: (BLUE, RED, BLUE), 4: (BLUE,) * 4,
    5: (BLUE, BLUE, RED, BLUE, BLUE), 6: (GREEN,) * 2 + (RED,) * 4,
    7: (GREEN,) * 3 + (RED,) * 4, 8: (BLUE,) * 8,
    9: (GREEN,) * 3 + (RED,) * 3 + (BLUE,) * 3,
}
DOT_RADII = {2: 10, 3: 8.5, 4: 8.5, 5: 8, 6: 8, 7: 6.5, 8: 6.5, 9: 6.5}
BAMBOO_POSITIONS = {
    2: [(32, 27), (32, 61)], 3: [(32, 22), (20, 63), (44, 63)],
    4: [(20, 26), (44, 26), (20, 62), (44, 62)],
    5: [(18, 22), (46, 22), (32, 44), (18, 66), (46, 66)],
    6: [(x, y) for y in (21, 44, 67) for x in (20, 44)],
    7: [(32, 15)] + [(x, y) for y in (34, 53, 72) for x in (20, 44)],
    9: [(x, y) for y in (20, 44, 68) for x in (17, 32, 47)],
}
# The eight has two outside stems and a central chevron in each half.
BAMBOO_EIGHT = ((15, 27, 0), (26, 27, 28), (38, 27, -28), (49, 27, 0),
                (15, 61, 0), (26, 61, -28), (38, 61, 28), (49, 61, 0))


def dot(x, y, color, radius=7):
    return (f'<g class="dot-mark" data-color="{color}" transform="translate({x} {y})">'
            f'<circle r="{radius}" fill="{color}"/>'
            f'<circle r="{radius-1.6}" fill="none" stroke="#fff9e9" stroke-width=".55"/>'
            '<path d="M0 0Q-4-4-4 0Q-4 4 0 0Q4 4 4 0Q4-4 0 0Z" fill="#fff9e9"/>'
            '</g>')


def bamboo(x, y, color=GREEN, *, height=20, angle=0):
    half = height / 2
    return (f'<g class="bamboo-mark" data-color="{color}" '
            f'transform="translate({x} {y}) rotate({angle})">'
            f'<path d="M-3.5 {-half}Q0 {-half+1.8} 3.5 {-half}'
            f'L2.6 {-half+3}V{half-3}L3.5 {half}'
            f'Q0 {half-1.8} -3.5 {half}L-2.6 {half-3}V{-half+3}Z" fill="{color}"/>'
            f'<path d="M-.6 {-half+3}V{half-3}" stroke="#fff9e9" stroke-width="1.1"/>'
            f'<path d="M-3.2 -1.5H3.2M-3.2 1.5H3.2" stroke="{color}" stroke-width="1.8"/>'
            '<path d="M-2.4 0H2.4" stroke="#fff9e9" stroke-width=".8"/>'
            '</g>')


def bird():
    return f'''<g class="bird-mark" stroke-linecap="round" stroke-linejoin="round">
<path d="M23 52Q15 66 12 76Q25 71 34 57M28 53Q28 66 26 76M33 52Q39 64 42 71" fill="none" stroke="{GREEN}" stroke-width="4"/>
<path d="M19 39Q15 52 27 57Q42 57 42 43Q35 35 36 30Q37 22 44 23Q51 25 46 33L43 38Q50 50 39 57Q27 65 17 53Q11 47 19 39Z" fill="{GREEN}"/>
<path d="M21 42Q20 53 32 54Q36 48 33 42Q26 38 21 42Z" fill="{BLUE}" stroke="#e7dabb" stroke-width="1.2"/>
<path d="M23 44L31 50M23 48L29 52" fill="none" stroke="#ede7c9" stroke-width="1"/>
<path d="M42 23L42 18M45 24L48 19" stroke="{RED}" stroke-width="2.5"/>
<path d="M48 28L54 31L48 33" fill="{RED}"/>
<circle cx="44" cy="28" r="2" fill="#fff9e9"/><circle cx="44.5" cy="28" r=".9" fill="#173627"/>
<path d="M34 57L33 63L29 65M40 55L41 63L46 64M18 78H48" fill="none" stroke="{RED}" stroke-width="1.6"/>
</g>'''


def face(kind, red=False):
    if kind >= 27:
        if kind == 31:
            return '<rect x="13" y="20" width="38" height="48" rx="3" fill="none" stroke="'+BLUE+'" stroke-width="3"/><rect x="17" y="24" width="30" height="40" rx="1" fill="none" stroke="'+BLUE+'" stroke-width=".8"/>'
        glyph = {27: '東', 28: '南', 29: '西', 30: '北', 32: '發', 33: '中'}[kind]
        color = GREEN if kind == 32 else RED if kind == 33 else '#202e28'
        return f'<text x="32" y="59" text-anchor="middle" font-family="Noto Serif CJK SC,Songti SC,serif" font-size="42" font-weight="700" fill="{color}">{glyph}</text>'
    rank, suit = kind % 9 + 1, kind // 9
    if suit == 0:
        digits = "一二三四五六七八九"
        return (f'<g text-anchor="middle" font-family="Noto Serif CJK SC,Songti SC,serif" font-weight="700">'
                f'<text x="32" y="38" font-size="29" fill="{RED if red else "#202e28"}">{digits[rank-1]}</text>'
                f'<text x="32" y="73" font-size="30" fill="{RED}">萬</text></g>')
    if suit == 1:
        if rank == 1:
            petals = ''.join(f'<ellipse cx="0" cy="-13" rx="3" ry="5.5" '
                             f'transform="rotate({angle})" fill="{GREEN}"/>' for angle in range(0, 360, 30))
            return (f'<g class="dot-mark" transform="translate(32 44)">'
                    f'<circle r="23" fill="none" stroke="{GREEN}" stroke-width="2"/>'
                    f'<circle r="20" fill="none" stroke="{GREEN}" stroke-width=".8"/>'
                    f'{petals}<circle r="7.2" fill="{RED}"/>'
                    '<circle r="3.2" fill="#fff9e9"/></g>')
        return ''.join(dot(x, y, RED if red else color, DOT_RADII[rank])
                       for (x, y), color in zip(DOT_POSITIONS[rank], DOT_COLORS[rank]))
    if rank == 1:
        return bird()
    if rank == 8:
        return ''.join(bamboo(x, y, height=25, angle=angle) for x, y, angle in BAMBOO_EIGHT)
    content = ''
    for index, (x, y) in enumerate(BAMBOO_POSITIONS[rank]):
        red_mark = red or rank == 5 and index == 2 or rank == 7 and index == 0 or rank == 9 and index % 3 == 1
        content += bamboo(x, y, RED if red_mark else GREEN, height=15 if rank == 7 else 19 if rank >= 6 else 22)
    return content


def svg(content, label):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 88" role="img" aria-label="{label}">
<title>{label}</title><defs><linearGradient id="ivory" x2=".3" y2="1"><stop stop-color="#fffef8"/><stop offset="1" stop-color="#eee9d8"/></linearGradient></defs>
<rect x="1" y="1" width="62" height="86" rx="6" fill="url(#ivory)" stroke="#d6d1bd"/>
<path d="M7 3h50q4 0 4 4v72" fill="none" stroke="#fff" opacity=".8"/>{content}</svg>'''


def main():
    tiles = ROOT / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    digits = "一二三四五六七八九"
    for kind in range(34):
        rank, suit = kind % 9 + 1, kind // 9
        content = face(kind)
        label = digits[rank - 1] + ("万", "筒", "条")[suit] if kind < 27 else ('东', '南', '西', '北', '白', '发', '中')[kind - 27]
        (tiles / f"{kind}.svg").write_text(svg(content, label), encoding="utf-8")
    for kind in (4, 13, 22):
        label = '赤五' + ('万', '筒', '条')[kind // 9]
        (tiles / f"{kind}-red.svg").write_text(svg(face(kind, red=True), label), encoding="utf-8")
    back = '<rect x="6" y="6" width="52" height="76" rx="4" fill="#2a7c63"/><rect x="10" y="10" width="44" height="68" rx="2" fill="none" stroke="#c1d2a8" opacity=".55"/><path d="M16 30l16-12 16 12-16 12zM16 56l16-12 16 12-16 12z" fill="none" stroke="#b8c99d" stroke-width="1.3" opacity=".55"/>'
    (tiles / "back.svg").write_text(svg(back, "麻将牌背"), encoding="utf-8")
    colors = (("#d9e6d5", "#2c614a", "#293b30"), ("#e8dfcb", "#8d7853", "#3d392b"),
              ("#dce3e7", "#567782", "#343d42"), ("#e8d9d0", "#925e4c", "#392f2b"))
    for index, (background, clothing, hair) in enumerate(colors):
        portrait = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80"><rect width="80" height="80" rx="40" fill="{background}"/><path d="M9 80q2-28 31-29 30 1 31 29" fill="{clothing}"/><path d="M34 48v11q6 7 12 0V48" fill="#d9a989"/><ellipse cx="40" cy="34" rx="16" ry="20" fill="#eac3a0"/><path d="M24 34q-6-27 17-27 21 0 16 28l-6-16q-12 9-24 6z" fill="{hair}"/><path d="M29 35h5m12 0h5" stroke="{hair}" stroke-width="2.2" stroke-linecap="round"/><path d="M36 45q4 3 8 0" fill="none" stroke="#a66b55" stroke-width="1.7" stroke-linecap="round"/><path d="M29 60l11 14 11-14" fill="none" stroke="#fff" opacity=".25" stroke-width="2"/></svg>'''
        (ROOT / f"avatar-{index}.svg").write_text(portrait, encoding="utf-8")
    (ROOT / "logo.svg").write_text('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><rect width="40" height="40" rx="11" fill="#1b5943"/><g fill="none" stroke="#e8d8ae" stroke-width="2"><path d="M11 12v16m9-20v24m9-20v16"/><path d="M8 17h6m3-4h6m3 10h6"/></g></svg>''', encoding="utf-8")
    print("Generated 38 tile SVGs, 4 portraits, and a logo.")


if __name__ == "__main__":
    main()
