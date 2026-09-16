"""Tkinter visualization of the eight-bit Jordan/ruler strategy."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from jordanchain import (
    JORDAN_CHAIN,
    MAX_INT256,
    coordinate_belief,
    jordan_move,
    physical_to_jordan,
    strategy_transition,
)
from util import int256_to_str


class WheelGameDemo(tk.Tk):
    CELL = 31
    GRID = 16

    def __init__(self) -> None:
        super().__init__()
        self.title("Eight-bit wheel: Jordan chain + ruler strategy")
        self.minsize(980, 680)

        self.step = 0
        self.playing = False
        self.after_id: str | None = None

        self._build_ui()
        self._draw_grid()
        self._render()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)

        title = ttk.Label(
            outer,
            text="Optimal blind search on the 8-node wheel",
            font=("TkDefaultFont", 18, "bold"),
        )
        title.pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Each square is one possible current state in Jordan coordinates. "
                "Move t uses a[ruler(t)]."
            ),
        ).pack(anchor="w", pady=(3, 12))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nw")
        self.canvas = tk.Canvas(
            left,
            width=self.CELL * self.GRID,
            height=self.CELL * self.GRID,
            background="#f4f6f8",
            highlightthickness=1,
            highlightbackground="#aeb7c2",
        )
        self.canvas.pack()

        legend = ttk.Frame(left)
        legend.pack(fill="x", pady=(8, 0))
        self._legend_item(legend, "#2979c9", "still possible").pack(side="left")
        self._legend_item(legend, "#e5e9ee", "not in current belief").pack(
            side="left", padx=(20, 0)
        )

        right = ttk.Frame(body, padding=(20, 0, 0, 0))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        self.headline = ttk.Label(right, font=("TkDefaultFont", 14, "bold"))
        self.headline.grid(row=0, column=0, sticky="w")

        stats = ttk.LabelFrame(right, text="Audited transition", padding=12)
        stats.grid(row=1, column=0, sticky="ew", pady=(10, 12))
        stats.columnconfigure(1, weight=1)
        self.stat_vars: dict[str, tk.StringVar] = {}
        rows = (
            ("move", "Move"),
            ("ruler", "Ruler level"),
            ("before", "Candidates before"),
            ("killed", "Winning pre-move state"),
            ("survivors", "Survivors after XOR"),
            ("rotation_added", "Candidates added by rotation"),
            ("after", "Candidates after rotation"),
            ("exact", "Exact F[d] → F[d−1] check"),
        )
        for row, (key, label) in enumerate(rows):
            ttk.Label(stats, text=label + ":").grid(
                row=row, column=0, sticky="w", padx=(0, 14), pady=2
            )
            variable = tk.StringVar()
            self.stat_vars[key] = variable
            ttk.Label(stats, textvariable=variable, font=("TkFixedFont", 11)).grid(
                row=row, column=1, sticky="w", pady=2
            )

        basis_text = "\n".join(
            f"a{k} = {int256_to_str(value)}  ({value:3d})"
            for k, value in enumerate(JORDAN_CHAIN)
        )
        basis = ttk.LabelFrame(right, text="Computed Jordan chain", padding=12)
        basis.grid(row=2, column=0, sticky="ew")
        ttk.Label(basis, text=basis_text, font=("TkFixedFont", 11)).pack(anchor="w")

        note = (
            "What “no resurrection” means here: after XOR removes the winning "
            "branch, taking all 8 possible rotations adds zero states. Raw state "
            "labels across different moves are relabelled by XOR, so those sets "
            "are not required to be nested."
        )
        ttk.Label(right, text=note, wraplength=390, foreground="#4f5b66").grid(
            row=3, column=0, sticky="ew", pady=(12, 0)
        )

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(14, 0))
        ttk.Button(controls, text="Reset", command=self.reset).pack(side="left")
        ttk.Button(controls, text="Previous", command=self.previous).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(controls, text="Next move", command=self.next).pack(
            side="left", padx=(8, 0)
        )
        self.play_button = ttk.Button(controls, text="Play", command=self.toggle_play)
        self.play_button.pack(side="left", padx=(8, 14))

        self.scale = ttk.Scale(
            controls,
            from_=0,
            to=MAX_INT256,
            orient="horizontal",
            command=self._scale_changed,
        )
        self.scale.pack(side="left", fill="x", expand=True)
        self.step_label = ttk.Label(controls, width=12, anchor="e")
        self.step_label.pack(side="right", padx=(12, 0))

    def _legend_item(self, parent: ttk.Frame, color: str, label: str) -> ttk.Frame:
        frame = ttk.Frame(parent)
        swatch = tk.Canvas(frame, width=14, height=14, highlightthickness=0)
        swatch.create_rectangle(1, 1, 13, 13, fill=color, outline=color)
        swatch.pack(side="left")
        ttk.Label(frame, text=label).pack(side="left", padx=(5, 0))
        return frame

    def _draw_grid(self) -> None:
        self.rectangles: list[int] = []
        self.labels: list[int] = []
        for coordinate in range(256):
            row, column = divmod(coordinate, self.GRID)
            x0, y0 = column * self.CELL, row * self.CELL
            rectangle = self.canvas.create_rectangle(
                x0,
                y0,
                x0 + self.CELL,
                y0 + self.CELL,
                fill="#e5e9ee",
                outline="#ffffff",
            )
            label = self.canvas.create_text(
                x0 + self.CELL / 2,
                y0 + self.CELL / 2,
                text=str(coordinate),
                font=("TkFixedFont", 8),
                fill="#7c8792",
            )
            self.rectangles.append(rectangle)
            self.labels.append(label)

    def _render(self) -> None:
        remaining = MAX_INT256 - self.step
        belief = coordinate_belief(remaining)
        for coordinate in range(256):
            active = coordinate in belief
            self.canvas.itemconfigure(
                self.rectangles[coordinate],
                fill="#2979c9" if active else "#e5e9ee",
            )
            self.canvas.itemconfigure(
                self.labels[coordinate],
                fill="#ffffff" if active else "#7c8792",
            )

        self.headline.configure(
            text=f"After {self.step} moves: {remaining} candidates remain"
        )
        self.step_label.configure(text=f"{self.step} / 255")
        if abs(float(self.scale.get()) - self.step) >= 0.5:
            self.scale.set(self.step)

        if self.step == 0:
            values = {
                "move": "—",
                "ruler": "—",
                "before": "255",
                "killed": "—",
                "survivors": "—",
                "rotation_added": "—",
                "after": "255",
                "exact": "ready",
            }
        else:
            transition = strategy_transition(self.step)
            move_coordinate = physical_to_jordan(transition.move)
            killed_coordinate = physical_to_jordan(transition.killed_pre_move)
            values = {
                "move": (
                    f"a{transition.ruler_level} = "
                    f"{int256_to_str(transition.move)} (physical), "
                    f"coord {move_coordinate}"
                ),
                "ruler": f"v2({self.step}) = {transition.ruler_level}",
                "before": str(len(transition.before)),
                "killed": (
                    f"{int256_to_str(transition.killed_pre_move)} "
                    f"(Jordan coord {killed_coordinate})"
                ),
                "survivors": str(len(transition.survivors_before_rotation)),
                "rotation_added": str(len(transition.rotation_added)),
                "after": str(len(transition.after_rotation)),
                "exact": "PASS" if transition.is_exact else "FAIL",
            }
        for key, value in values.items():
            self.stat_vars[key].set(value)

    def _scale_changed(self, raw_value: str) -> None:
        new_step = max(0, min(MAX_INT256, round(float(raw_value))))
        if new_step != self.step:
            self.step = new_step
            self._render()

    def next(self) -> None:
        if self.step < MAX_INT256:
            self.step += 1
            self._render()
        else:
            self._stop_playing()

    def previous(self) -> None:
        self._stop_playing()
        if self.step > 0:
            self.step -= 1
            self._render()

    def reset(self) -> None:
        self._stop_playing()
        self.step = 0
        self._render()

    def toggle_play(self) -> None:
        if self.playing:
            self._stop_playing()
            return
        if self.step == MAX_INT256:
            self.step = 0
        self.playing = True
        self.play_button.configure(text="Pause")
        self._play_tick()

    def _play_tick(self) -> None:
        if not self.playing:
            return
        self.next()
        if self.playing:
            self.after_id = self.after(90, self._play_tick)

    def _stop_playing(self) -> None:
        self.playing = False
        self.play_button.configure(text="Play")
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None


def main() -> None:
    app = WheelGameDemo()
    app.mainloop()


if __name__ == "__main__":
    main()
