import tkinter as tk

def startUI(featureNames, classNames):
    print('featureNames', featureNames)
    def train():
        learning_rate = eta_entry.get()
        num_epochs = epochs_entry.get()
        mse_threshold = mse_entry.get()
        print(f"Learning Rate (eta): {learning_rate}")
        print(f"Number of Epochs (m): {num_epochs}")
        print(f"MSE Threshold: {mse_threshold}")
        print(f"Add Bias: {bias_var.get()}")
        print(f"Selected Algorithm: {algorithm_var.get()}")

    root = tk.Tk()
    root.title("Single Perceptron Task 1")
    root.geometry("400x300")
    root.resizable(False, False)
    
    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack()
    
    # Combo Boxes
    combo_frame1 = tk.Frame(frame)
    combo_frame1.pack()
    
    combo_label1 = tk.Label(combo_frame1, text="Select X1")
    combo_label1.pack(side=tk.LEFT)
    combo_box1 = tk.StringVar()
    combo1 = tk.OptionMenu(combo_frame1, combo_box1, *featureNames)
    combo1.pack(side=tk.LEFT)
    
    combo_label2 = tk.Label(combo_frame1, text="Select X2")
    combo_label2.pack(side=tk.LEFT)
    combo_box2 = tk.StringVar()
    combo2 = tk.OptionMenu(combo_frame1, combo_box2, *featureNames)
    combo2.pack(side=tk.LEFT)
    
    # Spacer
    tk.Label(frame, text="").pack()
    
    combo_frame2 = tk.Frame(frame)
    combo_frame2.pack()
    
    combo_label3 = tk.Label(combo_frame2, text="Select Class 1")
    combo_label3.pack(side=tk.LEFT)
    combo_box3 = tk.StringVar(value=classNames[0])
    combo3 = tk.OptionMenu(combo_frame2, combo_box3, *classNames)
    combo3.pack(side=tk.LEFT)
    
    combo_label4 = tk.Label(combo_frame2, text="Select Class 2")
    combo_label4.pack(side=tk.LEFT)
    combo_box4 = tk.StringVar(value=classNames[1])
    combo4 = tk.OptionMenu(combo_frame2, combo_box4, *classNames)
    combo4.pack(side=tk.LEFT)
    
    # Text Fields
    eta_label = tk.Label(frame, text="Enter learning rate (eta):")
    eta_label.pack()
    eta_entry = tk.Entry(frame)
    eta_entry.pack()
    
    epochs_label = tk.Label(frame, text="Enter number of epochs (m):")
    epochs_label.pack()
    epochs_entry = tk.Entry(frame)
    epochs_entry.pack()
    
    mse_label = tk.Label(frame, text="Enter MSE threshold (mse_threshold):")
    mse_label.pack()
    mse_entry = tk.Entry(frame)
    mse_entry.pack()
    
    # Checkbox
    bias_var = tk.BooleanVar()
    bias_checkbox = tk.Checkbutton(frame, text="Add bias", variable=bias_var)
    bias_checkbox.pack()
    
    # Radio Buttons
    algorithm_var = tk.StringVar()
    algorithm_var.set("Perceptron")
    radio_frame = tk.Frame(frame)
    radio_frame.pack()

    
    perceptron_radio = tk.Radiobutton(radio_frame, text="Perceptron", variable=algorithm_var, value="Perceptron")
    perceptron_radio.pack(side=tk.LEFT)
    
    adaline_radio = tk.Radiobutton(radio_frame, text="Adaline", variable=algorithm_var, value="Adaline")
    adaline_radio.pack(side=tk.LEFT)
    
    # Submit Button
    train_button = tk.Button(root, text="Train", command=train)
    train_button.pack()
    
    root.mainloop()