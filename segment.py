from PIL import Image

def model_segment(path):
    image = Image.open(path)

    output = model.predict(image)

    polygons = output['polygon']

    return polygons