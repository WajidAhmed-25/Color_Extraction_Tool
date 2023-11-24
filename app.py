from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import os

app = Flask(__name__)

def convert_to_png(image_path):
    
    img = Image.open(image_path)

    png_path = os.path.splitext(image_path)[0] + ".png"

    img.save(png_path, "PNG")

    return png_path

def count_colors(hex_colors):
    return len(hex_colors)

def extract_colors(image_path, num_colors=10):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))

    unique_colors = np.unique(pixels, axis=0)

    if len(unique_colors) <= num_colors:
        # If the image contains fewer than or equal to num_colors, extract all unique colors
        hex_colors = ['#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2])) for color in unique_colors]
    else:
        # If the image contains more than num_colors, extract the prominent num_colors colors
        clf = KMeans(n_clusters=num_colors)
        labels = clf.fit_predict(unique_colors)
        center_colors = clf.cluster_centers_
        hex_colors = ['#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2])) for color in center_colors]

    return hex_colors


def replace_colors(image_path, num_colors, target_colors, replacement_colors):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    threshold = 50

    for i in range(num_colors):
        target_color_rgb = hex_to_rgb(target_colors[i])
        replacement_color_rgb = hex_to_rgb(replacement_colors[i])

        mask = np.all(np.abs(image - target_color_rgb) < threshold, axis=-1)
        image[mask] = replacement_color_rgb

    modified_image_path = os.path.splitext(image_path)[0] + "_modified.png"
    Image.fromarray(image).save(modified_image_path)

    return modified_image_path

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')


    if file:
        # Save the uploaded image
        image_path = f"static/{file.filename}"
        file.save(image_path)

        # Convert the image to PNG format
        png_path = convert_to_png(image_path)

        # Extract colors from the PNG image
        colors = extract_colors(png_path)
        color_count = count_colors(colors)

        return render_template('index.html', colors=colors, image_path=png_path, color_count=color_count)



@app.route('/replace_color', methods=['POST'])
def replace_color_route():
    try:
        num_colors = int(request.form['num_colors'])
        target_colors = [request.form[f'target_color_{i}'] for i in range(1, num_colors + 1)]
        replacement_colors = [request.form[f'replacement_color_{i}'] for i in range(1, num_colors + 1)]
        image_path = request.form['image_path']
    except KeyError:
        # Handle missing or incorrect form field names
        return render_template('index.html', error='Invalid form data for color replacement')

    modified_image_path = replace_colors(image_path, num_colors, target_colors, replacement_colors)
    return render_template('index.html', modified_image_path=modified_image_path)

if __name__ == '__main__':
    app.run(debug=True)
    
    
# Changes to change color

    
@app.route('/change_color', methods=['POST'])
def change_color():
    if request.method == 'POST':
        color = request.form.get('color')
        return render_template('change_color.html', color=color)   
    