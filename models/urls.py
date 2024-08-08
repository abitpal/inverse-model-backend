from django.urls import path


from . import views

urlpatterns = [
    path("layers", views.get_layers, name="get_layers"),
    path("graph", views.graph, name="get_graph"),
    path("check", views.checkConnection, name="check_connection"),
    path("color", views.get_color, name="get_color")
]