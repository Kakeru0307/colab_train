"""MIDI program / ViTex category の対応（方針 B: ギターは category 3）。"""

from __future__ import annotations

import muspy

# GM: Electric Guitar (clean) → program // 8 == 3
GUITAR_PROGRAM = 27

# パッチ category → DAW 再生用の代表 program（未指定は category * 8）
CATEGORY_DEFAULT_PROGRAM: dict[int, int] = {
    3: GUITAR_PROGRAM,
}


def program_for_category(category: int) -> int:
    return CATEGORY_DEFAULT_PROGRAM.get(category, category * 8)


def remap_tonal_program(music: muspy.Music, program: int = GUITAR_PROGRAM) -> muspy.Music:
    """非ドラムトラックの program を統一する（Guitar-TECHS 用）。"""
    for track in music.tracks:
        if not track.is_drum:
            track.program = program
    return music
