from predict import predict_speech

# Now, run a while loop to get user input and make predictions
while True:
    # Get user input
    jitter = float(input("\nEnter Jitter (%): "))
    shimmer = float(input("Enter Shimmer (%): "))
    wpm = float(input("Enter Words per Minute: "))
    pauses = float(input("Enter Pauses Duration (sec): "))

    # Pack the user input into a list (features)
    features = [jitter, shimmer, wpm, pauses]

    # Use the predict_speech function to get the prediction
    predicted_label = predict_speech(features)  # Predict the label directly using the function

    # Display the result
    print(f"Predicted Severity of Defect: {predicted_label}")

    # Ask if the user wants to make another prediction
    continue_input = input("\nDo you want to make another prediction? (y/n): ").lower()
    if continue_input != 'y':
        break

print("Goodbye!")
