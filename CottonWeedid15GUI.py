
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.applications import resnet50
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


# Parameters
# Load class names for weed types based on your training dataset
class_names = ['Carpetweeds', 'Crabgrass', 'Eclipta', 'Goosegrass', 'Morningglory',
    'Nutsedge', 'PalmerAmaranth', 'Prickly Sida', 'Purslane', 'Ragweed',
    'Sicklepod', 'SpottedSpurge', 'SpurredAnoda', 'Swinecress', 'Waterhemp']

# Load your trained models (update with actual file paths on your local machine)
vgg16_model_path = 'CottonWeed_VGG16.h5'  # Path to your VGG16 model
resnet50_model_path = 'CottonWeed_ResNet50.h5'  # Path to your ResNet50 model
mobilenet_model_path = 'CottonWeed_MobileNet.h5'
vgg16_SA_model_path = 'CottonWeed_VGG16_SoftAttention.h5'  # Path to your VGG16 model


# Load class names for weed types based on your training dataset
class_names = [
    'Carpetweeds', 'Crabgrass', 'Eclipta', 'Goosegrass', 'Morningglory',
    'Nutsedge', 'PalmerAmaranth', 'Prickly Sida', 'Purslane', 'Ragweed',
    'Sicklepod', 'SpottedSpurge', 'SpurredAnoda', 'Swinecress', 'Waterhemp'
]

# Load your trained models (update with actual file paths on your local machine)

vgg16_model_path = 'CottonWeed_VGG16.h5'  # Path to your VGG16 model
resnet50_model_path = 'CottonWeed_ResNet50.h5'  # Path to your ResNet50 model
mobilenet_model_path = 'CottonWeed_MobileNet.h5'
vgg16_SA_model_path = 'CottonWeed_VGG16_SoftAttention.h5'  # Path to your VGG16 model
# vgg16_model = tf.keras.models.load_model(vgg16_model_path)
# resnet50_model = tf.keras.models.load_model(resnet50_model_path)

# Define Tkinter window
root = tk.Tk()
root.title("Weed Detection")


# Function to upload an image
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        root.file_path = file_path  # Store file path for later use
        img = Image.open(file_path)
        img.thumbnail((250, 250))  # Resize for display
        img_display = ImageTk.PhotoImage(img)

        # Display image in Tkinter window
        image_label.config(image=img_display)
        image_label.image = img_display
        file_name_label.config(text=file_path)


# Function to analyze image with pretrained model
def analyze():
    try:
        model_name = selected_model.get()
        if not hasattr(root, 'file_path'):
            messagebox.showerror("Error", "Please upload an image first!")
            return

        if model_name == 'VGG16':
            model = tf.keras.models.load_model(vgg16_model_path)
        elif model_name == 'ResNet50':
            model = tf.keras.models.load_model(resnet50_model_path)
        elif model_name == 'MobileNet':
            model = tf.keras.models.load_model(mobilenet_model_path)
        elif model_name == 'VGG16_SoftAttention':
            model = tf.keras.models.load_model(vgg16_SA_model_path)
        else:
            messagebox.showerror("Error", "Unsupported model selected!")
    except Exception as e:
        messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    # Load and preprocess the image for VGG16
    img = Image.open(root.file_path)
    img = img.resize((512, 512))  # Resize to (512, 512) to match VGG16 input
    img_array = img_to_array(img) / 127.5 - 1.0  # Normalize for VGG16
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Perform prediction with VGG16

    prediction = model.predict(img_array)
    accuracy = np.max(prediction) * 100

    # Get the predicted class
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_class_index]

    print(predicted_class)

    # Display the results
    result_text.config(state=tk.NORMAL)
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, f"Analyzing {root.file_path} using VGG16...\n")
    result_text.insert(tk.END, f"Detection complete! Accuracy: {accuracy:.2f}%\n")
    result_text.insert(tk.END, f"Predicted Class: {predicted_class}\n")
    result_text.config(state=tk.DISABLED)

    # Show the prediction bar graph
    plot_prediction_graph(prediction[0])



# Function to plot prediction graph (bar chart)
def plot_prediction_graph(prediction):
    classes = class_names
    plt.figure(figsize=(10, 6))
    plt.barh(classes, prediction)
    plt.xlabel('Prediction Confidence')
    plt.title('Class Prediction Confidence')
    plt.xlim(0, 1)
    plt.tight_layout()

    # Save plot to a local file (on your machine)
    plt.savefig('prediction_graph.png')  # Saved to the current directory
    plt.close()

    # Display the graph in Tkinter
    img = Image.open('prediction_graph.png')
    img.thumbnail((350, 350))
    img_display = ImageTk.PhotoImage(img)
    graph_label.config(image=img_display)
    graph_label.image = img_display


# Define Tkinter components
frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

upload_button = tk.Button(frame, text="Upload Image", command=upload_image)
upload_button.grid(row=0, column=0, padx=10)

model_label = tk.Label(frame, text="Select Model:")
model_label.grid(row=1, column=0, padx=10)

selected_model = tk.StringVar()
vgg16_radiobutton = tk.Radiobutton(frame, text="VGG16", variable=selected_model, value="VGG16")
vgg16_radiobutton.grid(row=1, column=1, padx=10)

resnet_radiobutton = tk.Radiobutton(frame, text="ResNet50", variable=selected_model, value="ResNet50")
resnet_radiobutton.grid(row=1, column=2, padx=10)

mobilenet_radiobutton = tk.Radiobutton(frame, text="MobileNet", variable=selected_model, value="MobileNet")
mobilenet_radiobutton.grid(row=1, column=3, padx=10)

vgg16_SA_radiobutton = tk.Radiobutton(frame, text="VGG16_SoftAttention", variable=selected_model, value="VGG16_SoftAttention")
vgg16_SA_radiobutton.grid(row=1, column=4, padx=10)

analyze_button = tk.Button(frame, text="Analyze", command=analyze)
analyze_button.grid(row=2, column=0, columnspan=3, pady=20)

# Display Image and Result
image_label = tk.Label(root)
image_label.pack(pady=10)

file_name_label = tk.Label(root, text="")
file_name_label.pack()

result_text = tk.Text(root, height=6, width=50, wrap=tk.WORD, state=tk.DISABLED)
result_text.pack(pady=10)

graph_label = tk.Label(root)
graph_label.pack(pady=10)

# Start Tkinter loop
root.mainloop()
