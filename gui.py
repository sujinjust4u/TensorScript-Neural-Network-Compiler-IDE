import tkinter as tk
import sys
import io
import traceback
from compiler import compile_ts

class TensorScriptGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TensorScript IDE (Desktop Edition)")
        self.root.geometry("1000x800")
        
        self.root.configure(bg="#c0c0c0")
        
        self._build_menu()
        self._build_toolbar()
        self._build_panes()
        
        # Load sample
        self.editor.insert(tk.END, """model Classifier {\n  input shape(784)\n  layer dense(128, relu)\n  layer dropout(0.3)\n  layer dense(10, softmax)\n  loss categorical_crossentropy\n  optimizer adam(lr=0.001)\n  train epochs=5, batch=32\n}""")

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        compile_menu = tk.Menu(menubar, tearoff=0)
        compile_menu.add_command(label="Build Project", command=self.compile_code)
        menubar.add_cascade(label="Compile", menu=compile_menu)
        self.root.config(menu=menubar)

    def _build_toolbar(self):
        toolbar = tk.Frame(self.root, bd=2, relief=tk.RAISED, bg="#c0c0c0")
        toolbar.pack(side=tk.TOP, fill=tk.X)
        btn = tk.Button(toolbar, text="Compile Output", command=self.compile_code, highlightbackground="#c0c0c0", font=("Arial", 9, "bold"))
        btn.pack(side=tk.LEFT, padx=4, pady=2)

    def _build_panes(self):
        # We explicitly use tk.Frame and .place layout mechanics because macOS
        # frequently fails to render tk.PanedWindow boundaries dynamically correctly.
        container = tk.Frame(self.root, bg="#c0c0c0")
        container.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Left pane
        left_frame = tk.Frame(container, bg="#c0c0c0")
        left_frame.place(relx=0, rely=0, relwidth=0.5, relheight=1)
        
        left_inner = tk.Frame(left_frame, bd=2, relief=tk.SUNKEN)
        left_inner.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        header = tk.Label(left_inner, text=" Source Editor (model.ts)", bg="#000080", fg="white", font=("Arial", 10, "bold"), anchor="w")
        header.pack(fill=tk.X)
        
        self.editor = tk.Text(left_inner, font=("Courier", 14), undo=True)
        self.editor.pack(fill=tk.BOTH, expand=True)
        
        # Right pane
        right_frame = tk.Frame(container, bg="#c0c0c0")
        right_frame.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
        
        # We manually structure dual logical boxes instead of using ttk.Notebook 
        # protecting strictly against Mac Classic styling dropouts
        right_top = tk.Frame(right_frame, bd=2, relief=tk.SUNKEN)
        right_top.place(relx=0, rely=0, relwidth=1, relheight=0.5)
        
        tk.Label(right_top, text=" Trace Logs Window", bg="#000080", fg="white", font=("Arial", 10, "bold"), anchor="w").pack(fill=tk.X)
        self.logs_output = tk.Text(right_top, font=("Courier", 13), bg="black", fg="#c0c0c0")
        self.logs_output.pack(fill=tk.BOTH, expand=True)
        self.logs_output.insert(tk.END, "C:\\> Awaiting compiler instructions...\n")
        
        right_bot = tk.Frame(right_frame, bd=2, relief=tk.SUNKEN)
        right_bot.place(relx=0, rely=0.5, relwidth=1, relheight=0.5)
        
        tk.Label(right_bot, text=" Generated PyTorch (temp_gui_out.py)", bg="#000080", fg="white", font=("Arial", 10, "bold"), anchor="w").pack(fill=tk.X)
        self.py_output = tk.Text(right_bot, font=("Courier", 13), bg="black", fg="#55ff55")
        self.py_output.pack(fill=tk.BOTH, expand=True)
        self.py_output.insert(tk.END, "REM Auto-generated outputs will populate here.\n")

    def compile_code(self):
        code = self.editor.get("1.0", tk.END)
        with open('temp_gui.ts', 'w') as f:
            f.write(code)
            
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            compile_ts('temp_gui.ts', 'temp_gui_out.py')
            self.logs_output.delete("1.0", tk.END)
            self.logs_output.insert(tk.END, captured_output.getvalue())
            
            with open('temp_gui_out.py', 'r') as f:
                py_code = f.read()
            self.py_output.delete("1.0", tk.END)
            self.py_output.insert(tk.END, py_code)
        except Exception as e:
            self.logs_output.delete("1.0", tk.END)
            self.logs_output.insert(tk.END, captured_output.getvalue() + "\n\nCRITICAL ERROR:\n" + traceback.format_exc())
        finally:
            sys.stdout = old_stdout

if __name__ == "__main__":
    root = tk.Tk()
    app = TensorScriptGUI(root)
    root.mainloop()
