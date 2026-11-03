
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
    return (x - 2.0) ** 2


# ============================================================
# 2. Levy Flight：Mantegna 方法
# ============================================================

def levy_flight(beta=1.5):

    numerator = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
    )

    denominator = (
        math.gamma((1 + beta) / 2)
        * beta
        * 2 ** ((beta - 1) / 2)
    )

    sigma_u = (numerator / denominator) ** (1 / beta)

    u = np.random.normal(0, sigma_u)
    v = np.random.normal(0, 1)

    if abs(v) < 1e-12:
        v = 1e-12

    step = u / (abs(v) ** (1 / beta))

    return step


# ============================================================
# 3. Cuckoo Search
# ============================================================

class CuckooSearch:

    def __init__(self):

        # 搜尋範圍
        self.lower = -5.0
        self.upper = 5.0

        # CS 原始參數
        self.n_nests = 15
        self.pa = 0.25
        self.alpha = 0.25
        self.beta = 1.5

        # 修改為 50 次迭代
        self.max_iterations = 50

        # 初始化候選巢穴
        self.nests = np.random.uniform(
            self.lower,
            self.upper,
            self.n_nests
        )

        self.fitness = objective_function(self.nests)

        # 初始化最佳解
        best_index = np.argmin(self.fitness)

        self.best_x = float(self.nests[best_index])
        self.best_f = float(self.fitness[best_index])

        # 迭代次數
        self.iteration = 0

        # 歷史紀錄：包含初始解
        self.best_history = [self.best_f]
        self.position_history = [self.best_x]

        # Levy Flight 步長
        self.step_history = []

        # 每次產生的新候選解
        self.new_solution_history = []

        # 表格歷史資料
        self.error_history = [abs(self.best_f)]

    # ========================================================
    # 執行一次搜尋
    # ========================================================

    def step(self):

        if self.iteration >= self.max_iterations:
            return False

        # ----------------------------------------------------
        # A. 隨機選擇一個巢穴
        # ----------------------------------------------------

        nest_index = np.random.randint(0, self.n_nests)

        current_x = float(self.nests[nest_index])

        # ----------------------------------------------------
        # B. Levy Flight 產生新候選解
        # ----------------------------------------------------

        step_size = levy_flight(self.beta)

        new_x = current_x + self.alpha * step_size

        # 限制搜尋範圍
        new_x = np.clip(
            new_x,
            self.lower,
            self.upper
        )

        new_f = float(objective_function(new_x))

        # ----------------------------------------------------
        # C. 隨機選擇另一個巢穴比較
        # ----------------------------------------------------

        compare_index = np.random.randint(0, self.n_nests)

        compare_f = self.fitness[compare_index]

        # 新候選解較好時，更新巢穴
        if new_f < compare_f:

            self.nests[compare_index] = new_x
            self.fitness[compare_index] = new_f

        # ----------------------------------------------------
        # D. 巢穴淘汰
        # ----------------------------------------------------

        n_abandon = max(
            1,
            int(self.pa * self.n_nests)
        )

        # 找出函數值由大到小的巢穴
        worst_indices = np.argsort(self.fitness)[::-1]

        replaced = 0

        for index in worst_indices:

            # 保留目前最佳巢穴
            if self.fitness[index] == np.min(self.fitness):
                continue

            # 產生隨機巢穴
            random_x = np.random.uniform(
                self.lower,
                self.upper
            )

            random_f = float(objective_function(random_x))

            self.nests[index] = random_x
            self.fitness[index] = random_f

            replaced += 1

            if replaced >= n_abandon:
                break

        # ----------------------------------------------------
        # E. 更新目前最佳解
        # ----------------------------------------------------

        current_best_index = np.argmin(self.fitness)

        current_best_x = float(
            self.nests[current_best_index]
        )

        current_best_f = float(
            self.fitness[current_best_index]
        )

        # ----------------------------------------------------
        # F. 保留全域最佳解
        # ----------------------------------------------------

        if current_best_f < self.best_f:

            self.best_x = current_best_x
            self.best_f = current_best_f

        # ----------------------------------------------------
        # G. 記錄歷程
        # ----------------------------------------------------

        self.iteration += 1

        self.best_history.append(self.best_f)

        self.position_history.append(self.best_x)

        self.step_history.append(step_size)

        self.new_solution_history.append(new_x)

        # 誤差：與理論最佳函數值 0 的差距
        error = abs(self.best_f - 0.0)

        self.error_history.append(error)

        return True


# ============================================================
# 4. Tkinter 動態視窗
# ============================================================

class CuckooSearchGUI:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Cuckoo Search Optimization - 50 Iterations"
        )

        self.root.geometry("1450x900")

        # 初始化演算法
        self.cs = CuckooSearch()

        self.running = False
        self.after_id = None

        # 動態更新速度
        self.animation_delay = 100

        # ----------------------------------------------------
        # 控制區
        # ----------------------------------------------------

        control_frame = ttk.Frame(root)

        control_frame.pack(
            side=tk.TOP,
            fill=tk.X,
            padx=10,
            pady=8
        )

        ttk.Button(
            control_frame,
            text="開始",
            command=self.start
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame,
            text="暫停",
            command=self.pause
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame,
            text="重置",
            command=self.reset
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame,
            text="單步",
            command=self.single_step
        ).pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(
            control_frame,
            text=""
        )

        self.status_label.pack(
            side=tk.LEFT,
            padx=15
        )

        # ----------------------------------------------------
        # 主畫面：左側四張圖，右側表格
        # ----------------------------------------------------

        main_frame = ttk.Frame(root)

        main_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=5
        )

        # 左側圖表
        graph_frame = ttk.Frame(main_frame)

        graph_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        # 右側資料
        table_frame = ttk.LabelFrame(
            main_frame,
            text="迭代紀錄"
        )

        table_frame.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=False,
            padx=10
        )

        # ----------------------------------------------------
        # 建立四張圖
        # ----------------------------------------------------

        self.fig = Figure(
            figsize=(10, 8),
            dpi=100
        )

        self.ax1 = self.fig.add_subplot(221)
        self.ax2 = self.fig.add_subplot(222)
        self.ax3 = self.fig.add_subplot(223)
        self.ax4 = self.fig.add_subplot(224)

        self.fig.tight_layout(pad=3)

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=graph_frame
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        # ----------------------------------------------------
        # 右側表格
        # ----------------------------------------------------

        columns = (
            "iteration",
            "best_x",
            "best_f",
            "error"
        )

        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=30
        )

        self.tree.heading(
            "iteration",
            text="迭代次數"
        )

        self.tree.heading(
            "best_x",
            text="最佳位置"
        )

        self.tree.heading(
            "best_f",
            text="最佳函數值"
        )

        self.tree.heading(
            "error",
            text="誤差"
        )

        self.tree.column(
            "iteration",
            width=75,
            anchor=tk.CENTER
        )

        self.tree.column(
            "best_x",
            width=105,
            anchor=tk.CENTER
        )

        self.tree.column(
            "best_f",
            width=120,
            anchor=tk.CENTER
        )

        self.tree.column(
            "error",
            width=110,
            anchor=tk.CENTER
        )

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient=tk.VERTICAL,
            command=self.tree.yview
        )

        self.tree.configure(
            yscrollcommand=scrollbar.set
        )

        scrollbar.pack(
            side=tk.RIGHT,
            fill=tk.Y
        )

        self.tree.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        # 初始資料：第 0 次
        self.add_table_row(
            0,
            self.cs.best_x,
            self.cs.best_f,
            abs(self.cs.best_f)
        )

        self.update_plots()

        # 關閉視窗時停止排程
        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )

    # ========================================================
    # 5. 表格紀錄
    # ========================================================

    def add_table_row(self, iteration, best_x, best_f, error):

        self.tree.insert(
            "",
            tk.END,
            values=(
                iteration,
                f"{best_x:.6f}",
                f"{best_f:.10f}",
                f"{error:.10f}"
            )
        )

        # 捲動到最新紀錄
        self.tree.yview_moveto(1)

    # ========================================================
    # 6. 更新四張圖
    # ========================================================

    def update_plots(self):

        cs = self.cs

        # ----------------------------------------------------
        # Figure 1：目標函數與候選巢穴
        # ----------------------------------------------------

        self.ax1.clear()

        x = np.linspace(
            cs.lower,
            cs.upper,
            500
        )

        y = objective_function(x)

        self.ax1.plot(
            x,
            y,
            label="f(x) = (x - 2)^2"
        )

        self.ax1.scatter(
            cs.nests,
            cs.fitness,
            s=35,
            label="Candidate nests"
        )

        self.ax1.axvline(
            2,
            linestyle="--",
            label="Theoretical optimum x=2"
        )

        self.ax1.scatter(
            [cs.best_x],
            [cs.best_f],
            marker="*",
            s=180,
            label="Best solution"
        )

        if cs.new_solution_history:

            last_x = cs.new_solution_history[-1]

            self.ax1.scatter(
                [last_x],
                [objective_function(last_x)],
                marker="x",
                s=80,
                label="New candidate"
            )

        self.ax1.set_title("CS Optimization")

        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("f(x)")

        self.ax1.set_xlim(
            cs.lower,
            cs.upper
        )

        self.ax1.grid(True)
        self.ax1.legend(fontsize=8)

        # ----------------------------------------------------
        # Figure 2：最佳搜尋位置
        # ----------------------------------------------------

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

        self.ax2.set_title("CS Search Position")

        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Position")

        self.ax2.set_xlim(
            0,
            cs.max_iterations
        )

        self.ax2.grid(True)
        self.ax2.legend(fontsize=8)

        # ----------------------------------------------------
        # Figure 3：Levy Flight 步長
        # ----------------------------------------------------

        self.ax3.clear()

        if cs.step_history:

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

        self.ax3.set_title("Levy Flight Step Size")

        self.ax3.set_xlabel("Iteration")
        self.ax3.set_ylabel("Step Size")

        self.ax3.set_xlim(
            0,
            cs.max_iterations
        )

        self.ax3.grid(True)

        # ----------------------------------------------------
        # Figure 4：最佳函數值收斂曲線
        # ----------------------------------------------------

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

        self.ax4.set_title("CS Convergence Curve")

        self.ax4.set_xlabel("Iteration")
        self.ax4.set_ylabel("Best Function Value")

        self.ax4.set_xlim(
            0,
            cs.max_iterations
        )

        self.ax4.grid(True)
        self.ax4.legend(fontsize=8)

        # ----------------------------------------------------
        # 更新狀態列
        # ----------------------------------------------------

        self.status_label.config(
            text=(
                f"Iteration: {cs.iteration}/"
                f"{cs.max_iterations}    "
                f"Best x: {cs.best_x:.6f}    "
                f"Best f(x): {cs.best_f:.10f}    "
                f"Error: {abs(cs.best_f):.10f}"
            )
        )

        self.canvas.draw_idle()

    # ========================================================
    # 7. 執行一次迭代並更新表格
    # ========================================================

    def perform_step(self):

        success = self.cs.step()

        if success:

            self.update_plots()

            self.add_table_row(
                self.cs.iteration,
                self.cs.best_x,
                self.cs.best_f,
                abs(self.cs.best_f)
            )

        return success

    # ========================================================
    # 8. Start
    # ========================================================

    def start(self):

        if self.running:
            return

        if self.cs.iteration >= self.cs.max_iterations:
            return

        self.running = True

        self.animate()

    # ========================================================
    # 9. Pause
    # ========================================================

    def pause(self):

        self.running = False

        if self.after_id is not None:

            try:
                self.root.after_cancel(self.after_id)

            except tk.TclError:
                pass

            self.after_id = None

    # ========================================================
    # 10. 動態動畫
    # ========================================================

    def animate(self):

        if not self.running:
            return

        if self.cs.iteration >= self.cs.max_iterations:

            self.pause()

            return

        self.perform_step()

        # 達到 50 次後停止
        if self.cs.iteration >= self.cs.max_iterations:

            self.running = False

            self.after_id = None

            return

        if self.running:

            self.after_id = self.root.after(
                self.animation_delay,
                self.animate
            )

    # ========================================================
    # 11. 單步執行
    # ========================================================

    def single_step(self):

        self.pause()

        if self.cs.iteration < self.cs.max_iterations:

            self.perform_step()

    # ========================================================
    # 12. Reset
    # ========================================================

    def reset(self):

        self.pause()

        self.cs = CuckooSearch()

        # 清空表格
        for item in self.tree.get_children():

            self.tree.delete(item)

        # 加入第 0 次初始紀錄
        self.add_table_row(
            0,
            self.cs.best_x,
            self.cs.best_f,
            abs(self.cs.best_f)
        )

        self.update_plots()

    # ========================================================
    # 13. 關閉
    # ========================================================

    def close(self):

        self.pause()

        self.root.destroy()


# ============================================================
# 14. 執行程式
# ============================================================

if __name__ == "__main__":

    np.random.seed(7)

    root = tk.Tk()

    app = CuckooSearchGUI(root)

    root.mainloop()


