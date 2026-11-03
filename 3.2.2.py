
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import math

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ============================================================
# 1. 焊接梁成本函數
# ============================================================

def welded_beam_cost(w, L, h, d):
    """
    焊接梁成本函數

    w : 焊接厚度
    L : 焊接長度
    h : 梁高度
    d : 梁厚度
    """

    return (
        1.10471 * w**2 * L
        + 0.04811 * d * h * (14 + L)
    )


# ============================================================
# 2. Levy Flight
# ============================================================

def levy_flight(dim, alpha=0.03, lam=1.5):
    """
    使用 Mantegna 方法產生 Levy Flight 步長
    """

    sigma_u = (
        math.gamma(1 + lam)
        * math.sin(math.pi * lam / 2)
        /
        (
            math.gamma((1 + lam) / 2)
            * lam
            * 2 ** ((lam - 1) / 2)
        )
    ) ** (1 / lam)

    u = np.random.normal(
        0,
        sigma_u,
        dim
    )

    v = np.random.normal(
        0,
        1,
        dim
    )

    step = u / (
        np.abs(v) ** (1 / lam) + 1e-12
    )

    return alpha * step


# ============================================================
# 3. Cuckoo Search
# ============================================================

class CuckooSearch:

    def __init__(
        self,
        n=25,
        max_iter=100,
        pa=0.25,
        alpha=0.03,
        lam=1.5
    ):

        self.n = n
        self.max_iter = max_iter

        self.pa = pa
        self.alpha = alpha
        self.lam = lam

        # 四個變數下限
        self.lower = np.array([
            0.1,
            0.1,
            0.1,
            0.1
        ])

        # 四個變數上限
        self.upper = np.array([
            2.0,
            10.0,
            2.0,
            10.0
        ])

        self.reset()

    # --------------------------------------------------------
    # 初始化
    # --------------------------------------------------------

    def reset(self):

        self.iteration = 0

        # 初始化巢穴
        self.nests = np.random.uniform(
            self.lower,
            self.upper,
            size=(self.n, 4)
        )

        # 計算所有巢穴成本
        self.fitness = np.array([
            self.evaluate(nest)
            for nest in self.nests
        ])

        # 初始最佳解
        best_index = np.argmin(
            self.fitness
        )

        self.best_solution = (
            self.nests[best_index].copy()
        )

        self.best_cost = float(
            self.fitness[best_index]
        )

        # 最佳成本歷史
        self.cost_history = [
            self.best_cost
        ]

        # 最佳變數歷史
        self.variable_history = [
            self.best_solution.copy()
        ]

        # 初始巢穴歷史
        self.nest_history = [
            self.nests.copy()
        ]

        # 資料表紀錄
        self.records = []

        self.records.append({
            "iteration": 0,
            "w": self.best_solution[0],
            "L": self.best_solution[1],
            "h": self.best_solution[2],
            "d": self.best_solution[3],
            "cost": self.best_cost,
            "improvement": None
        })

    # --------------------------------------------------------
    # 成本計算
    # --------------------------------------------------------

    def evaluate(self, solution):

        w, L, h, d = solution

        return float(
            welded_beam_cost(
                w, L, h, d
            )
        )

    # --------------------------------------------------------
    # 執行一代
    # --------------------------------------------------------

    def step(self):

        if self.iteration >= self.max_iter:

            return False

        previous_best = self.best_cost

        # ====================================================
        # A. Levy Flight 搜尋
        # ====================================================

        for i in range(self.n):

            current = self.nests[i].copy()

            # Levy Flight
            step_size = levy_flight(
                4,
                alpha=self.alpha,
                lam=self.lam
            )

            # 產生新解
            new_solution = (
                current
                + step_size
                * (current - self.best_solution)
            )

            # 限制變數範圍
            new_solution = np.clip(
                new_solution,
                self.lower,
                self.upper
            )

            new_cost = self.evaluate(
                new_solution
            )

            # 隨機選擇一個巢穴
            j = np.random.randint(
                0,
                self.n
            )

            # 新解較好則取代
            if new_cost < self.fitness[j]:

                self.nests[j] = (
                    new_solution.copy()
                )

                self.fitness[j] = new_cost

        # ====================================================
        # B. 淘汰部分巢穴
        # ====================================================

        abandon_count = max(
            1,
            int(self.pa * self.n)
        )

        worst_indices = np.argsort(
            self.fitness
        )[-abandon_count:]

        for index in worst_indices:

            new_nest = np.random.uniform(
                self.lower,
                self.upper
            )

            self.nests[index] = (
                new_nest.copy()
            )

            self.fitness[index] = (
                self.evaluate(new_nest)
            )

        # ====================================================
        # C. 更新最佳解
        # ====================================================

        best_index = np.argmin(
            self.fitness
        )

        current_best_cost = float(
            self.fitness[best_index]
        )

        current_best_solution = (
            self.nests[best_index].copy()
        )

        if current_best_cost < self.best_cost:

            self.best_cost = (
                current_best_cost
            )

            self.best_solution = (
                current_best_solution.copy()
            )

        # ====================================================
        # D. 保存歷史
        # ====================================================

        improvement = max(
            0.0,
            previous_best - self.best_cost
        )

        self.iteration += 1

        self.cost_history.append(
            self.best_cost
        )

        self.variable_history.append(
            self.best_solution.copy()
        )

        # 保存每一代所有巢穴位置
        self.nest_history.append(
            self.nests.copy()
        )

        # 保存資料表
        self.records.append({
            "iteration": self.iteration,
            "w": self.best_solution[0],
            "L": self.best_solution[1],
            "h": self.best_solution[2],
            "d": self.best_solution[3],
            "cost": self.best_cost,
            "improvement": improvement
        })

        return True


# ============================================================
# 4. Tkinter GUI
# ============================================================

class CuckooSearchGUI:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Cuckoo Search 焊接梁最佳化"
        )

        self.root.geometry(
            "1600x950"
        )

        self.root.minsize(
            1100,
            700
        )

        # CS 參數
        self.max_iter = 100

        self.cs = CuckooSearch(
            n=25,
            max_iter=self.max_iter,
            pa=0.25,
            alpha=0.03,
            lam=1.5
        )

        # 自動執行狀態
        self.running = False

        # Tkinter 排程
        self.after_id = None

        # 建立 GUI
        self.create_widgets()

        # 初始畫面
        self.update_plot()

    # ========================================================
    # 建立 GUI
    # ========================================================

    def create_widgets(self):

        # ----------------------------------------------------
        # 上方控制區
        # ----------------------------------------------------

        control_frame = ttk.LabelFrame(
            self.root,
            text="演算法控制"
        )

        control_frame.pack(
            side=tk.TOP,
            fill=tk.X,
            padx=10,
            pady=8
        )

        # 開始
        self.start_button = ttk.Button(
            control_frame,
            text="開始",
            command=self.start
        )

        self.start_button.pack(
            side=tk.LEFT,
            padx=5,
            pady=8
        )

        # 暫停
        self.pause_button = ttk.Button(
            control_frame,
            text="暫停",
            command=self.pause
        )

        self.pause_button.pack(
            side=tk.LEFT,
            padx=5,
            pady=8
        )

        # 重置
        self.reset_button = ttk.Button(
            control_frame,
            text="重置",
            command=self.reset
        )

        self.reset_button.pack(
            side=tk.LEFT,
            padx=5,
            pady=8
        )

        # 單步
        self.step_button = ttk.Button(
            control_frame,
            text="單步執行",
            command=self.single_step
        )

        self.step_button.pack(
            side=tk.LEFT,
            padx=5,
            pady=8
        )

        # 狀態
        self.status_label = ttk.Label(
            control_frame,
            text="等待開始",
            font=("Microsoft JhengHei", 10)
        )

        self.status_label.pack(
            side=tk.LEFT,
            padx=10
        )

        # 最佳成本
        self.cost_label = ttk.Label(
            control_frame,
            text="最佳成本：",
            font=("Microsoft JhengHei", 10)
        )

        self.cost_label.pack(
            side=tk.LEFT,
            padx=10
        )

        # 最佳變數
        self.variable_label = ttk.Label(
            control_frame,
            text="",
            font=("Microsoft JhengHei", 10)
        )

        self.variable_label.pack(
            side=tk.LEFT,
            padx=10
        )

        # ----------------------------------------------------
        # 主畫面
        # ----------------------------------------------------

        main_frame = ttk.Frame(
            self.root
        )

        main_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=5
        )

        # 左側圖表
        self.plot_frame = ttk.Frame(
            main_frame
        )

        self.plot_frame.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        # 右側資料表
        self.table_frame = ttk.LabelFrame(
            main_frame,
            text="每代最佳解資料表"
        )

        self.table_frame.pack(
            side=tk.RIGHT,
            fill=tk.Y,
            padx=5
        )

        # 建立圖表
        self.create_plot()

        # 建立資料表
        self.create_table()

    # ========================================================
    # 建立圖表
    # ========================================================

    def create_plot(self):

        self.fig = Figure(
            figsize=(11, 8),
            dpi=100
        )

        # ----------------------------------------------------
        # 3D 成本曲面
        # ----------------------------------------------------

        self.ax1 = self.fig.add_subplot(
            221,
            projection="3d"
        )

        self.ax1.set_title(
            "3D CS Search Animation"
        )

        self.ax1.set_xlabel("w")
        self.ax1.set_ylabel("L")
        self.ax1.set_zlabel("Cost")

        # ----------------------------------------------------
        # 收斂曲線
        # ----------------------------------------------------

        self.ax2 = self.fig.add_subplot(
            222
        )

        self.ax2.set_title(
            "Convergence Curve"
        )

        self.ax2.set_xlabel(
            "Iteration"
        )

        self.ax2.set_ylabel(
            "Best Cost"
        )

        self.ax2.grid(
            True
        )

        # ----------------------------------------------------
        # 四變數演化
        # ----------------------------------------------------

        self.ax3 = self.fig.add_subplot(
            212
        )

        self.ax3.set_title(
            "Four Variables Evolution"
        )

        self.ax3.set_xlabel(
            "Iteration"
        )

        self.ax3.set_ylabel(
            "Variable Value"
        )

        self.ax3.grid(
            True
        )

        self.fig.tight_layout()

        # ----------------------------------------------------
        # 放入 Tkinter
        # ----------------------------------------------------

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=self.plot_frame
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

    # ========================================================
    # 建立資料表
    # ========================================================

    def create_table(self):

        columns = (
            "iteration",
            "w",
            "L",
            "h",
            "d",
            "cost",
            "improvement"
        )

        self.tree = ttk.Treeview(
            self.table_frame,
            columns=columns,
            show="headings",
            height=30
        )

        headings = {
            "iteration": "代數",
            "w": "w",
            "L": "L",
            "h": "h",
            "d": "d",
            "cost": "最佳成本",
            "improvement": "成本改善"
        }

        for column in columns:

            self.tree.heading(
                column,
                text=headings[column]
            )

            self.tree.column(
                column,
                width=90,
                anchor=tk.CENTER
            )

        self.tree.column(
            "iteration",
            width=55
        )

        # 捲軸
        scrollbar = ttk.Scrollbar(
            self.table_frame,
            orient=tk.VERTICAL,
            command=self.tree.yview
        )

        self.tree.configure(
            yscrollcommand=scrollbar.set
        )

        self.tree.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        scrollbar.pack(
            side=tk.RIGHT,
            fill=tk.Y
        )

    # ========================================================
    # 更新資料表
    # ========================================================

    def update_table(self):

        # 清除舊資料
        for item in self.tree.get_children():

            self.tree.delete(item)

        # 插入最新紀錄
        for record in self.cs.records:

            improvement = record["improvement"]

            if improvement is None:

                improvement_text = "-"

            else:

                improvement_text = (
                    f"{improvement:.8f}"
                )

            self.tree.insert(
                "",
                tk.END,
                values=(
                    record["iteration"],
                    f'{record["w"]:.6f}',
                    f'{record["L"]:.6f}',
                    f'{record["h"]:.6f}',
                    f'{record["d"]:.6f}',
                    f'{record["cost"]:.8f}',
                    improvement_text
                )
            )

        # 移至最後一列
        children = self.tree.get_children()

        if children:

            self.tree.see(
                children[-1]
            )

    # ========================================================
    # 3D 成本曲面與動畫
    # ========================================================

    def draw_surface(self):

        # 清除上一代畫面
        self.ax1.clear()

        # ----------------------------------------------------
        # A. 建立成本曲面
        # ----------------------------------------------------

        w_values = np.linspace(
            0.1,
            2.0,
            40
        )

        L_values = np.linspace(
            0.1,
            10.0,
            40
        )

        W, LL = np.meshgrid(
            w_values,
            L_values
        )

        # 使用目前最佳解的 h 和 d
        best_h = self.cs.best_solution[2]

        best_d = self.cs.best_solution[3]

        Z = welded_beam_cost(
            W,
            LL,
            best_h,
            best_d
        )

        # 畫曲面
        self.ax1.plot_surface(
            W,
            LL,
            Z,
            cmap="viridis",
            alpha=0.45,
            edgecolor="none"
        )

        # ----------------------------------------------------
        # B. 取得目前巢穴位置
        # ----------------------------------------------------

        nests = self.cs.nests

        nest_w = nests[:, 0]

        nest_L = nests[:, 1]

        nest_h = nests[:, 2]

        nest_d = nests[:, 3]

        # 計算每個巢穴的成本
        nest_cost = np.array([
            welded_beam_cost(
                w,
                L,
                h,
                d
            )
            for w, L, h, d in nests
        ])

        # ----------------------------------------------------
        # C. 畫出所有巢穴
        # ----------------------------------------------------

        self.ax1.scatter(
            nest_w,
            nest_L,
            nest_cost,
            color="green",
            s=35,
            alpha=0.8,
            label="Nests"
        )

        # ----------------------------------------------------
        # D. 畫目前最佳解
        # ----------------------------------------------------

        best_w = self.cs.best_solution[0]

        best_L = self.cs.best_solution[1]

        best_cost = self.cs.best_cost

        self.ax1.scatter(
            best_w,
            best_L,
            best_cost,
            color="red",
            s=180,
            marker="*",
            edgecolors="black",
            label="Best Solution"
        )

        # ----------------------------------------------------
        # E. 畫最佳解軌跡
        # ----------------------------------------------------

        history = np.array(
            self.cs.variable_history
        )

        path_w = history[:, 0]

        path_L = history[:, 1]

        path_cost = np.array([
            welded_beam_cost(
                row[0],
                row[1],
                row[2],
                row[3]
            )
            for row in history
        ])

        self.ax1.plot(
            path_w,
            path_L,
            path_cost,
            color="red",
            linewidth=2,
            marker=".",
            markersize=3,
            label="Best Path"
        )

        # ----------------------------------------------------
        # F. 設定三維圖
        # ----------------------------------------------------

        self.ax1.set_title(
            f"3D CS Search Animation\n"
            f"Iteration: {self.cs.iteration}\n"
            f"h={best_h:.4f}, d={best_d:.4f}"
        )

        self.ax1.set_xlabel(
            "w"
        )

        self.ax1.set_ylabel(
            "L"
        )

        self.ax1.set_zlabel(
            "Cost"
        )

        self.ax1.set_xlim(
            0.1,
            2.0
        )

        self.ax1.set_ylim(
            0.1,
            10.0
        )

        self.ax1.legend(
            loc="upper left"
        )

    # ========================================================
    # 收斂曲線
    # ========================================================

    def draw_convergence(self):

        self.ax2.clear()

        iterations = np.arange(
            len(self.cs.cost_history)
        )

        self.ax2.plot(
            iterations,
            self.cs.cost_history,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="Best Cost"
        )

        self.ax2.set_title(
            "Convergence Curve"
        )

        self.ax2.set_xlabel(
            "Iteration"
        )

        self.ax2.set_ylabel(
            "Best Cost"
        )

        self.ax2.grid(
            True
        )

        self.ax2.legend()

    # ========================================================
    # 四變數演化圖
    # ========================================================

    def draw_variables(self):

        self.ax3.clear()

        history = np.array(
            self.cs.variable_history
        )

        iterations = np.arange(
            len(history)
        )

        # w
        self.ax3.plot(
            iterations,
            history[:, 0],
            label="w"
        )

        # L
        self.ax3.plot(
            iterations,
            history[:, 1],
            label="L"
        )

        # h
        self.ax3.plot(
            iterations,
            history[:, 2],
            label="h"
        )

        # d
        self.ax3.plot(
            iterations,
            history[:, 3],
            label="d"
        )

        self.ax3.set_title(
            "Four Variables Evolution"
        )

        self.ax3.set_xlabel(
            "Iteration"
        )

        self.ax3.set_ylabel(
            "Variable Value"
        )

        self.ax3.grid(
            True
        )

        self.ax3.legend()

    # ========================================================
    # 更新全部畫面
    # ========================================================

    def update_plot(self):

        # 更新 3D 動畫
        self.draw_surface()

        # 更新收斂圖
        self.draw_convergence()

        # 更新四變數圖
        self.draw_variables()

        # 更新資料表
        self.update_table()

        # 更新狀態
        self.update_status()

        # 重新排版
        self.fig.tight_layout()

        # 更新畫布
        self.canvas.draw_idle()

    # ========================================================
    # 更新狀態文字
    # ========================================================

    def update_status(self):

        self.cost_label.config(
            text=(
                f"最佳成本："
                f"{self.cs.best_cost:.8f}"
            )
        )

        w, L, h, d = self.cs.best_solution

        self.variable_label.config(
            text=(
                f"w={w:.4f}  "
                f"L={L:.4f}  "
                f"h={h:.4f}  "
                f"d={d:.4f}"
            )
        )

        if self.running:

            status = "正在執行"

        elif self.cs.iteration >= self.max_iter:

            status = "已完成"

        else:

            status = "已暫停"

        self.status_label.config(
            text=(
                f"{status} | "
                f"第 {self.cs.iteration} / "
                f"{self.max_iter} 代"
            )
        )

    # ========================================================
    # 開始
    # ========================================================

    def start(self):

        # 已完成
        if self.cs.iteration >= self.max_iter:

            messagebox.showinfo(
                "提示",
                "演算法已完成，請按重置重新開始。"
            )

            return

        # 避免重複啟動
        if self.running:

            return

        self.running = True

        self.update_status()

        # 開始動畫迭代
        self.run_iteration()

    # ========================================================
    # 自動迭代
    # ========================================================

    def run_iteration(self):

        # 已暫停
        if not self.running:

            return

        # 已完成
        if self.cs.iteration >= self.max_iter:

            self.running = False

            self.after_id = None

            self.update_plot()

            messagebox.showinfo(
                "完成",
                "Cuckoo Search 最佳化已完成！"
            )

            return

        # 執行一代
        success = self.cs.step()

        if success:

            # 更新圖表、巢穴、最佳解、資料表
            self.update_plot()

        # 排程下一代
        if self.running:

            self.after_id = self.root.after(
                200,
                self.run_iteration
            )

    # ========================================================
    # 暫停
    # ========================================================

    def pause(self):

        self.running = False

        # 取消自動排程
        if self.after_id is not None:

            try:

                self.root.after_cancel(
                    self.after_id
                )

            except tk.TclError:

                pass

            self.after_id = None

        self.update_status()

    # ========================================================
    # 重置
    # ========================================================

    def reset(self):

        # 停止動畫
        self.pause()

        # 重新初始化 CS
        self.cs.reset()

        # 更新畫面
        self.update_plot()

        self.update_status()

    # ========================================================
    # 單步執行
    # ========================================================

    def single_step(self):

        # 先暫停自動執行
        self.pause()

        # 已完成
        if self.cs.iteration >= self.max_iter:

            messagebox.showinfo(
                "提示",
                "已達到最大迭代次數。"
            )

            return

        # 執行一代
        self.cs.step()

        # 更新畫面
        self.update_plot()

        self.update_status()


# ============================================================
# 5. 主程式
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = CuckooSearchGUI(
        root
    )

    root.mainloop()


