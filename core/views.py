from django.shortcuts import render
from django.http import JsonResponse
from .utils.ml.predict import predict_speech
from .utils.speech.features import extract_features

def manual_entry(request):
    if request.method == "POST":
        try:
            jitter = float(request.POST.get("jitter", 0))
            shimmer = float(request.POST.get("shimmer", 0))
            speech_rate = float(request.POST.get("speech_rate", 0))
            pause = float(request.POST.get("pause", 0))

            features = [jitter, shimmer, speech_rate, pause]
            prediction = predict_speech(features)

            return JsonResponse({"prediction": prediction})
        except Exception as e:
            return JsonResponse({"error": str(e)})

    return render(request, "core/manual_entry.html")

def upload_audio(request):
    if request.method == "POST":
        try:
            audio_file = request.FILES["audio"]

            # Extract features directly from the uploaded file
            speech_features = extract_features(audio_file, input="file")
            
            # Convert non-serializable types to native Python types
            def convert_to_serializable(value):
                if isinstance(value, (float, int, str)):
                    return value  # Already serializable
                try:
                    return float(value)  # Try converting to float if it's a non-serializable number
                except Exception:
                    return str(value)  # Convert to string if it's something else (like a custom object)

            # Apply conversion to all values in the speech_features dictionary
            speech_features_serializable = {key: convert_to_serializable(value)
                                            for key, value in speech_features.items()}

            features = [
                speech_features_serializable["jitter"],
                speech_features_serializable["shimmer"],
                speech_features_serializable["speech_rate"],
                speech_features_serializable["pause_duration"]
            ]
            
            prediction = predict_speech(features)

            return JsonResponse({"prediction": prediction, "speech_features": speech_features_serializable})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return render(request, "core/upload_audio.html")
