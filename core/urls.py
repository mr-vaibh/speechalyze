from django.urls import path
from .views import manual_entry, upload_audio

urlpatterns = [
    path("", upload_audio, name="upload_audio"),
    path("manual-entry/", manual_entry, name="manual_entry"),
]
