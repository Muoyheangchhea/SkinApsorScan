import tensorflow as tf

# Load the skin model from SavedModel format and save it as .h5
print("Converting skin_model...")
model = tf.keras.models.load_model('./models/skin_model')  # Path to skin_model folder
model.save('./models/skin_model.h5')  # Save the model in .h5 format
print("skin_model converted and saved as ./models/skin_model.h5")

# Load the acne model from SavedModel format and save it as .h5
print("Converting acne_model...")
acne_model = tf.keras.models.load_model('./models/acne_model')  # Path to acne_model folder
acne_model.save('./models/acne_model.h5')  # Save the model in .h5 format
print("acne_model converted and saved as ./models/acne_model.h5")
