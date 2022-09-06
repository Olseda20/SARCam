# Search and Rescue Drone Camera System
My dissertation project displaying my work to calibrate image data and fuse visual + thermal images together to provide clearer image data for the purposes of search and rescue. Additional some Saliencey and CNN models were used to attempt imporving these images with the intent of allowing autonomous detection of people within the images.

## Camera Configuration
This camera system will consist a Visual + Thermal camera module. Setup onto a Raspberry Pi.
The Visual Camera is a standard PiCamera 
The Thermal Camera is a Seek Thermal Compact Pro Camera with a modified USB cable attatched to a Rapsberry Pi.
https://www.amazon.co.uk/Seek-Thermal-Compact-Resolution-Imaging-Black/dp/B01NBU1AVN

## Goal
The overall goal of this system is to analyse areas and locate victims lost that may be lost
in the scottish mountains. 
The Visual and Thermal camera imagery were fused together to generate clearer spots on images allowing for easier identification of people. 
Additionally, the visual cameras allowed CNN and Saliency techniques to be used to identify people within the image, the thermal cameras can be utilised for warm people in cold regions as well as using this data for similar saliency and ML methods.

Saliency Models

<img width="597" alt="image" src="https://user-images.githubusercontent.com/49950899/188759407-17ff7f78-eb7e-4587-ac34-37d335e4f4c2.png">

Visual + Thermal Image Fusion 

<img width="592" alt="image" src="https://user-images.githubusercontent.com/49950899/188759470-c1414de0-1fe4-47ac-953e-3d60e807335a.png">


Library used to interface the thermal camera with the Raspberry Pi.
[ Seek-thermal-Python ](https://github.com/LaboratoireMecaniqueLille/Seek-thermal-Python/blob/master/seekpro.py ) - Python code to capture images from the thermal camera 
