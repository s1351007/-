

# ============================================================
# 2.5.2 程式主題：CS 搜尋流程示意
#
# Cuckoo Search + Tkinter Dynamic Animation
#
# Objective:
#       f(x) = (x - 2)^2
#
# Search range:
#       -5 <= x <= 5
#
# Theoretical optimum:
#       x* = 2
#       f(x*) = 0
#
# Parameters:
#       n_nests = 20
#       alpha = 0.1
#       pa = 0.25
#       beta = 1.5
#       max_iterations = 100
#
# GUI:
#       Start / Pause / Reset / Step
#
# Figures:
#       1. Objective Function
#       2. Best Position
#       3. Levy Flight Step Size
#       4. Convergence Curve
# ============================================================


import tkinter as tk
from tkinter import ttk

import math
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# 1. 目標函數
# ============================================================

def objective_function(x):

    """
    目標函數：

        f(x) = (x - 2)^2

    理論最佳位置：
        x = 2

    理論最佳函數值：
        f(x) = 0
    """

    return (x - 2.0) ** 2


# ============================================================
# 2. Levy Flight
# ============================================================

def levy_flight(beta=1.5):

    """
    使用 Mantegna 方法產生 Levy Flight 隨機步長。

    beta = 1.5

    輸出：
        一個具有長尾特性的隨機步長
    """

    numerator = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
    )

    denominator = (
        math.gamma((1 + beta) / 2)
        * beta
        * 2 ** ((beta - 1) / 2)
    )

    sigma_u = (
        numerator / denominator
    ) ** (1 / beta)

    u = np.random.normal(
        0,
        sigma_u
    )

    v = np.random.normal(
        0,
        1
    )

    # 避免除以零
    if abs(v) < 1e-12:

        v = 1e-12

    step = u / (
        abs(v) ** (1 / beta)
    )

    return step


# ============================================================
# 3. Cuckoo Search 演算法
# ============================================================

class CuckooSearch:

    def __init__(self):

        # ----------------------------------------------------
        # 搜尋範圍
        # ----------------------------------------------------

        self.lower = -5.0

        self.upper = 5.0

        # ----------------------------------------------------
        # CS 參數
        # ----------------------------------------------------

        self.n_nests = 20

        self.alpha = 0.1

        self.pa = 0.25

        self.beta = 1.5

        self.max_iterations = 100

        # ----------------------------------------------------
        # 初始化候選巢穴
        # ----------------------------------------------------

        self.nests = np.random.uniform(
            self.lower,
            self.upper,
            self.n_nests
        )

        self.fitness = objective_function(
            self.nests
        )

        # ----------------------------------------------------
        # 初始化最佳解
        # ----------------------------------------------------

        best_index = np.argmin(
            self.fitness
        )

        self.best_x = float(
            self.nests[best_index]
        )

        self.best_f = float(
            self.fitness[best_index]
        )

        # ----------------------------------------------------
        # 搜尋歷程
        # ----------------------------------------------------

        self.iteration = 0

        self.best_history = [
            self.best_f
        ]

        self.position_history = [
            self.best_x
        ]

        self.step_history = []

        self.candidate_history = []

    # ========================================================
    # 執行一次搜尋
    # ========================================================

    def step(self):

        if self.iteration >= self.max_iterations:

            return False

        # ----------------------------------------------------
        # A. 選擇目前巢穴
        # ----------------------------------------------------

        nest_index = np.random.randint(
            0,
            self.n_nests
        )

        current_x = float(
            self.nests[nest_index]
        )

        # ----------------------------------------------------
        # B. Levy Flight
        # ----------------------------------------------------

        step_size = levy_flight(
            self.beta
        )

        new_x = (
            current_x
            + self.alpha * step_size
        )

        # ----------------------------------------------------
        # C. 限制搜尋範圍
        # ----------------------------------------------------

        new_x = np.clip(
            new_x,
            self.lower,
            self.upper
        )

        new_f = float(
            objective_function(new_x)
        )

        # ----------------------------------------------------
        # D. 隨機選擇比較巢穴
        # ----------------------------------------------------

        compare_index = np.random.randint(
            0,
            self.n_nests
        )

        compare_f = self.fitness[
            compare_index
        ]

        # ----------------------------------------------------
        # E. 新候選解較佳時更新巢穴
        # ----------------------------------------------------

        if new_f < compare_f:

            self.nests[
                compare_index
            ] = new_x

            self.fitness[
                compare_index
            ] = new_f

        # ----------------------------------------------------
        # F. 棄置部分巢穴
        # ----------------------------------------------------

        n_abandon = max(
            1,
            int(self.pa * self.n_nests)
        )

        # 找出函數值較差的巢穴
        worst_indices = np.argsort(
            self.fitness
        )[::-1]

        replaced = 0

        for index in worst_indices:

            # 保留目前最佳巢穴
            if self.fitness[index] == np.min(
                self.fitness
            ):

                continue

            # 隨機產生新的候選巢穴
            random_x = np.random.uniform(
                self.lower,
                self.upper
            )

            random_f = float(
                objective_function(random_x)
            )

            self.nests[index] = random_x

            self.fitness[index] = random_f

            replaced += 1

            if replaced >= n_abandon:

                break

        # ----------------------------------------------------
        # G. 找出目前最佳候選解
        # ----------------------------------------------------

        current_best_index = np.argmin(
            self.fitness
        )

        current_best_x = float(
            self.nests[current_best_index]
        )

        current_best_f = float(
            self.fitness[current_best_index]
        )

        # ----------------------------------------------------
        # H. 更新全域最佳解
        # ----------------------------------------------------

        if current_best_f < self.best_f:

            self.best_x = current_best_x

            self.best_f = current_best_f

        # ----------------------------------------------------
        # I. 記錄搜尋歷程
        # ----------------------------------------------------

        self.iteration += 1

        self.best_history.append(
            self.best_f
        )

        self.position_history.append(
            self.best_x
        )

        self.step_history.append(
            step_size
        )

        self.candidate_history.append(
            new_x
        )

        return True


# ============================================================
# 4. Tkinter 動態視窗
# ============================================================

class CuckooSearchGUI:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "CS Optimization - Section 2.5.2"
        )

        self.root.geometry(
            "1400x900"
        )

        # ----------------------------------------------------
        # CS 初始化
        # ----------------------------------------------------

        self.cs = CuckooSearch()

        self.running = False

        self.after_id = None

        self.delay = 100

        # ----------------------------------------------------
        # 控制區
        # ----------------------------------------------------

        control_frame = ttk.Frame(
            root
        )

        control_frame.pack(
            side=tk.TOP,
            fill=tk.X,
            padx=10,
            pady=8
        )

        ttk.Button(
            control_frame,
            text="Start",
            command=self.start
        ).pack(
            side=tk.LEFT,
            padx=5
        )

        ttk.Button(
            control_frame,
            text="Pause",
            command=self.pause
        ).pack(
            side=tk.LEFT,
            padx=5
        )

        ttk.Button(
            control_frame,
            text="Reset",
            command=self.reset
        ).pack(
            side=tk.LEFT,
            padx=5
        )

        ttk.Button(
            control_frame,
            text="Step",
            command=self.single_step
        ).pack(
            side=tk.LEFT,
            padx=5
        )

        # ----------------------------------------------------
        # 搜尋資訊
        # ----------------------------------------------------

        self.status_label = ttk.Label(
            control_frame,
            text=""
        )

        self.status_label.pack(
            side=tk.LEFT,
            padx=15
        )

        # ----------------------------------------------------
        # 建立 Matplotlib Figure
        # ----------------------------------------------------

        self.fig = Figure(
            figsize=(13, 8),
            dpi=100
        )

        self.ax1 = self.fig.add_subplot(
            221
        )

        self.ax2 = self.fig.add_subplot(
            222
        )

        self.ax3 = self.fig.add_subplot(
            223
        )

        self.ax4 = self.fig.add_subplot(
            224
        )

        self.fig.tight_layout(
            pad=3
        )

        # ----------------------------------------------------
        # 嵌入 Tkinter
        # ----------------------------------------------------

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=root
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        # ----------------------------------------------------
        # 初始圖表
        # ----------------------------------------------------

        self.update_plots()

    # ========================================================
    # 5. 更新四張圖
    # ========================================================

    def update_plots(self):

        cs = self.cs

        # ====================================================
        # Figure 1
        # 目標函數與候選巢穴
        # ====================================================

        self.ax1.clear()

        x = np.linspace(
            cs.lower,
            cs.upper,
            500
        )

        y = objective_function(
            x
        )

        self.ax1.plot(
            x,
            y,
            label="f(x) = (x - 2)^2"
        )

        # 候選巢穴
        self.ax1.scatter(
            cs.nests,
            cs.fitness,
            s=35,
            label="Candidate nests"
        )

        # 理論最佳解
        self.ax1.axvline(
            2,
            linestyle="--",
            label="Theoretical optimum x = 2"
        )

        # 最佳解
        self.ax1.scatter(
            [cs.best_x],
            [cs.best_f],
            marker="*",
            s=180,
            label="Best solution"
        )

        # 最新候選解
        if len(cs.candidate_history) > 0:

            latest_x = cs.candidate_history[-1]

            self.ax1.scatter(
                [latest_x],
                [objective_function(latest_x)],
                marker="x",
                s=80,
                label="New candidate"
            )

        self.ax1.set_title(
            "CS Optimization"
        )

        self.ax1.set_xlabel(
            "x"
        )

        self.ax1.set_ylabel(
            "f(x)"
        )

        self.ax1.set_xlim(
            cs.lower,
            cs.upper
        )

        self.ax1.grid(
            True
        )

        self.ax1.legend(
            fontsize=8
        )

        # ====================================================
        # Figure 2
        # 最佳搜尋位置
        # ====================================================

        self.ax2.clear()

        iterations = np.arange(
            len(cs.position_history)
        )

        self.ax2.plot(
            iterations,
            cs.position_history,
            marker=".",
            markersize=3,
            label="Best position"
        )

        self.ax2.axhline(
            2,
            linestyle="--",
            label="Theoretical optimum"
        )

        self.ax2.set_title(
            "CS Search Position"
        )

        self.ax2.set_xlabel(
            "Iteration"
        )

        self.ax2.set_ylabel(
            "Position"
        )

        self.ax2.set_xlim(
            0,
            cs.max_iterations
        )

        self.ax2.grid(
            True
        )

        self.ax2.legend(
            fontsize=8
        )

        # ====================================================
        # Figure 3
        # Levy Flight 步長
        # ====================================================

        self.ax3.clear()

        if len(cs.step_history) > 0:

            step_iterations = np.arange(
                1,
                len(cs.step_history) + 1
            )

            self.ax3.plot(
                step_iterations,
                cs.step_history,
                marker=".",
                markersize=3
            )

            self.ax3.axhline(
                0,
                linestyle="--"
            )

        self.ax3.set_title(
            "Levy Flight Step Size"
        )

        self.ax3.set_xlabel(
            "Iteration"
        )

        self.ax3.set_ylabel(
            "Step Size"
        )

        self.ax3.set_xlim(
            0,
            cs.max_iterations
        )

        self.ax3.grid(
            True
        )

        # ====================================================
        # Figure 4
        # 收斂曲線
        # ====================================================

        self.ax4.clear()

        iterations = np.arange(
            len(cs.best_history)
        )

        self.ax4.plot(
            iterations,
            cs.best_history,
            marker=".",
            markersize=3,
            label="Best fitness"
        )

        self.ax4.axhline(
            0,
            linestyle="--",
            label="Theoretical optimum"
        )

        self.ax4.set_title(
            "CS Convergence Curve"
        )

        self.ax4.set_xlabel(
            "Iteration"
        )

        self.ax4.set_ylabel(
            "Best Function Value"
        )

        self.ax4.set_xlim(
            0,
            cs.max_iterations
        )

        self.ax4.grid(
            True
        )

        self.ax4.legend(
            fontsize=8
        )

        # ====================================================
        # 更新資訊列
        # ====================================================

        self.status_label.config(
            text=(
                f"Iteration: {cs.iteration}/"
                f"{cs.max_iterations}    "
                f"Best x: {cs.best_x:.6f}    "
                f"Best f(x): {cs.best_f:.8f}"
            )
        )

        # ====================================================
        # 更新 Tkinter 畫面
        # ====================================================

        self.canvas.draw()

    # ========================================================
    # 6. Start
    # ========================================================

    def start(self):

        if self.running:

            return

        if self.cs.iteration >= self.cs.max_iterations:

            return

        self.running = True

        self.animate()

    # ========================================================
    # 7. Pause
    # ========================================================

    def pause(self):

        self.running = False

        if self.after_id is not None:

            try:

                self.root.after_cancel(
                    self.after_id
                )

            except tk.TclError:

                pass

            self.after_id = None

    # ========================================================
    # 8. 動態搜尋動畫
    # ========================================================

    def animate(self):

        if not self.running:

            return

        if self.cs.iteration >= self.cs.max_iterations:

            self.pause()

            return

        # 執行一次 CS 搜尋
        success = self.cs.step()

        if success:

            # 更新所有圖表
            self.update_plots()

        # 安排下一次動畫
        if self.running:

            self.after_id = self.root.after(
                self.delay,
                self.animate
            )

    # ========================================================
    # 9. Step
    # ========================================================

    def single_step(self):

        self.pause()

        if self.cs.iteration < self.cs.max_iterations:

            self.cs.step()

            self.update_plots()

    # ========================================================
    # 10. Reset
    # ========================================================

    def reset(self):

        self.pause()

        self.cs = CuckooSearch()

        self.update_plots()

    # ========================================================
    # 11. 關閉視窗
    # ========================================================

    def close(self):

        self.pause()

        self.root.destroy()


# ============================================================
# 12. 主程式
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = CuckooSearchGUI(
        root
    )

    root.protocol(
        "WM_DELETE_WINDOW",
        app.close
    )

    root.mainloop()
