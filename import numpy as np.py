import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk  # Moved ImageTk import here
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Load and prepare MNIST data
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
x_train = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
y_train = tf.keras.utils.to_categorical(y_train_full)
y_test = tf.keras.utils.to_categorical(y_test)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15
)
datagen.fit(x_train)

# Define residual block
def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

# Build and compile model
def create_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    x = residual_block(x, 32)
    x = layers.MaxPooling2D(2)(x)
    x = residual_block(x, 64)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Load or train model with error handling
def load_or_train_model():
    model_path = 'mnist_model.h5'
    try:
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)
        else:
            model = create_model()
            print("Training new model...")
            model.fit(datagen.flow(x_train, y_train, batch_size=32),
                      epochs=10,
                      validation_data=(x_test, y_test))
            model.save(model_path)
            print(f"Model saved to {model_path}")
            return model
    except Exception as e:
        print(f"Error loading or training model: {e}")
        return None

# GUI Application
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition Application")
        self.root.geometry("900x600")
        self.root.configure(bg='#f5f5f5')
        
        # Set theme and style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#f5f5f5')
        style.configure('TButton', font=('Helvetica', 12))
        style.configure('TLabel', font=('Helvetica', 12), background='#f5f5f5')

        # Load model
        self.model = load_or_train_model()
        if self.model is None:
            raise RuntimeError("Failed to initialize model")

        # Main container - split into left and right panels
        self.main_container = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Drawing area
        self.left_frame = ttk.Frame(self.main_container)
        self.main_container.add(self.left_frame, weight=1)
        
        # Right panel - Results and visualization
        self.right_frame = ttk.Frame(self.main_container)
        self.main_container.add(self.right_frame, weight=1)
        
        # ===== Left Panel Contents =====
        ttk.Label(self.left_frame, text="Draw a Digit", font=('Helvetica', 16, 'bold')).pack(pady=10)
        
        # Canvas for drawing
        self.canvas_frame = ttk.Frame(self.left_frame)
        self.canvas_frame.pack(pady=10)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=300, height=300, bg='white',
                                highlightthickness=2, highlightbackground='#3498db')
        self.canvas.pack()

        # Create image for drawing
        self.image = Image.new('L', (300, 300), 255)
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Button frame
        self.button_frame = ttk.Frame(self.left_frame)
        self.button_frame.pack(pady=20)
        
        # Create custom styled buttons
        predict_btn = tk.Button(self.button_frame, text="Predict", 
                                command=self.predict, bg='#2ecc71', fg='white',
                                font=('Helvetica', 12, 'bold'), padx=15, pady=8,
                                relief=tk.RAISED, bd=0)
        predict_btn.pack(side=tk.LEFT, padx=10)
        
        clear_btn = tk.Button(self.button_frame, text="Clear", 
                              command=self.clear, bg='#e74c3c', fg='white',
                              font=('Helvetica', 12, 'bold'), padx=15, pady=8,
                              relief=tk.RAISED, bd=0)
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        # ===== Right Panel Contents =====
        ttk.Label(self.right_frame, text="Results", font=('Helvetica', 16, 'bold')).pack(pady=10)
        
        # Result display frame
        self.result_display = ttk.Frame(self.right_frame)
        self.result_display.pack(pady=10, fill=tk.X)
        
        # Large prediction display
        self.prediction_frame = ttk.Frame(self.result_display)
        self.prediction_frame.pack(pady=5)
        
        self.result_label = ttk.Label(self.prediction_frame, text="?", 
                                      font=('Helvetica', 72, 'bold'))
        self.result_label.pack()
        
        self.confidence_label = ttk.Label(self.prediction_frame, 
                                          text="Confidence: 0%",
                                          font=('Helvetica', 14))
        self.confidence_label.pack()
        
        # Visualization frame
        self.viz_frame = ttk.Frame(self.right_frame)
        self.viz_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        ttk.Label(self.viz_frame, text="Probability Chart", font=('Helvetica', 14)).pack(pady=5)
        
        # Create matplotlib figure for visualization
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas_viz = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_viz.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty visualization
        self.update_visualization([0]*10)
        
        # Status variables
        self.last_x = None
        self.last_y = None
        
        # Show preview of processed image
        self.preview_frame = ttk.Frame(self.right_frame)
        self.preview_frame.pack(pady=10)
        
        ttk.Label(self.preview_frame, text="Processed Input Image", font=('Helvetica', 12)).pack(pady=5)
        
        self.preview_canvas = tk.Canvas(self.preview_frame, width=112, height=112, bg='black')
        self.preview_canvas.pack()

    def draw_line(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=20, fill='black', capstyle=tk.ROUND,
                                    smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y],
                           fill=0, width=20)
        self.last_x = x
        self.last_y = y

    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (300, 300), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="?")
        self.confidence_label.config(text="Confidence: 0%")
        self.update_visualization([0]*10)
        self.preview_canvas.delete('all')

    def update_visualization(self, probabilities):
        # Clear previous plot
        self.ax.clear()
        
        # Create bar chart
        digits = list(range(10))
        bars = self.ax.bar(digits, probabilities, color='#3498db')
        
        # Highlight the highest probability
        if max(probabilities) > 0:
            max_index = np.argmax(probabilities)
            bars[max_index].set_color('#2ecc71')
        
        # Customize plot
        self.ax.set_title('Digit Probabilities', fontsize=12)
        self.ax.set_xlabel('Digit', fontsize=10)
        self.ax.set_ylabel('Probability', fontsize=10)
        self.ax.set_xticks(digits)
        self.ax.set_ylim([0, 1.05])
        self.ax.set_xlim([-0.5, 9.5])
        self.ax.grid(axis='y', linestyle='--', alpha=0.7)  # Fixed this line
        
        # Add value labels
        for i, prob in enumerate(probabilities):
            if prob > 0.05:  # Only show significant values
                self.ax.text(i, prob + 0.02, f'{prob:.2f}', ha='center', fontsize=8)
                
        # Redraw canvas
        self.fig.tight_layout()
        self.canvas_viz.draw()

    def show_preview(self, img_array):
        # Scale up to show the 28x28 image more clearly (4x)
        preview_img = Image.fromarray((img_array[0] * 255).astype(np.uint8).reshape(28, 28))
        preview_img = preview_img.resize((112, 112), Image.NEAREST)
        
        # Convert to PhotoImage for tkinter
        self.preview_photo = ImageTk.PhotoImage(preview_img)
        self.preview_canvas.create_image(56, 56, image=self.preview_photo)

    def predict(self):
        try:
            # Process image
            img = self.image.resize((28, 28))
            img_array = np.array(img).astype('float32') / 255.0
            img_array = 1.0 - img_array  # Invert colors
            img_array = img_array.reshape(1, 28, 28, 1)
            
            # Show processed preview
            self.show_preview(img_array)

            # Make prediction
            prediction = self.model.predict(img_array)
            digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            # Update visualization
            self.update_visualization(prediction[0])

            # Update UI
            self.result_label.config(text=f"{digit}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
        except Exception as e:
            print(f"Error during prediction: {e}")
            self.result_label.config(text="Error")
            self.confidence_label.config(text="Confidence: N/A")

def main():
    try:
        root = tk.Tk()
        app = DigitRecognizerApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error in application: {e}")

if __name__ == "__main__":
    main()