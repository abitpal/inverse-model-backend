from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse

from rest_framework.decorators import api_view

import tensorflow #using tensorflow and keras 2.14.0
import keras
import numpy as np

from models.inverse_model import InverseModelDense

def save_uploaded_models(f):
    with open("models/static/model.keras", "wb") as t: 
        t.write(f.getbuffer())

def save_uploaded_data(f):
    with open("models/static/data.npy", "wb") as t:
        t.write(f.getbuffer())


def save_uploaded_color(f):
    with open("models/static/color.npy", "wb") as t:
        t.write(f.getbuffer())

@api_view(["GET"])
def checkConnection(request):
    return HttpResponse("Inverse Model Backend is up and running!")


@api_view(["POST"])
def get_layers(request):
    data = request.FILES
    model_file = data["model"].file #this returns a bytesIO object
    save_uploaded_models(model_file)
    model = tensorflow.keras.saving.load_model("models/static/model.keras")
    layers = model.layers
    target_layer_name = request.data["layer"]
    target_layer = None

    for layer in layers:
        if (layer.name == target_layer_name):
            target_layer = layer

    found = True
    if (target_layer == None):
        found = False

    ret = JsonResponse({
        "found": found,
    })
    return ret

@api_view(["POST"])
def graph(request):

    print("Received Graph Request")

    files = request.FILES

    model_file = files["model"].file #this returns a bytesIO object
    data_file = files["data"].file
    target_layer_name = request.data["layer"]

    save_uploaded_models(model_file)
    save_uploaded_data(data_file)


    #establish the model and the data
    model = tensorflow.keras.saving.load_model("models/static/model.keras")
    data = np.load("models/static/data.npy")

    for layer in model.layers:
        if (layer.name == target_layer_name):
            target_layer = layer


    print("Found target layer")
    IMD = InverseModelDense(model, target_layer)

    print("Sampling data now...")
    samp = IMD.sample(data)[3] #getting the third dimension

    print("Finished Sampling data...")

    sampX = samp[:, 0].tolist()
    sampY = samp[:, 1].tolist()
    sampZ = samp[:, 2].tolist()

    try:
        color_file = files["color"].file
        save_uploaded_color(color_file)
        color = np.load("models/static/color.npy")
        return JsonResponse({
            "x": sampX,
            "y": sampY,
            "z": sampZ,
            "hasColor": True,
            "color": color.tolist()
        })
    except:
        pass


    return JsonResponse({
        "x": sampX,
        "y": sampY,
        "z": sampZ,
        "hasColor": False,
    })
    







