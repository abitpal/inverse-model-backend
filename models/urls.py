from django.urls import path


from . import views

urlpatterns = [
    path("layers", views.get_layers, name="get_layers"),
    path("graph", views.graph, name="get_graph"),
]