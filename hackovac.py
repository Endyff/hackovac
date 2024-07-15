import streamlit as st
import cv2
import numpy as np
import fpdf

# Load the image
image = cv2.imread('cat03.jpg')
image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png', 'webp'])
if image_file is not None:
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)

# User settings
# TODO: Change description to czech, modify value ranges (possibly add fill-in-boxes rather than sliders)
threshold = st.sidebar.slider('Threshold', 0, 255, 15)
blur_kernel_size = st.sidebar.slider('blur', 0, 120, 2) 
show_grid = st.sidebar.checkbox('Show Grid', False)
invert = st.sidebar.checkbox('Invert', False)
width_size = st.sidebar.slider('Width of final image', 1, 100, 10)
show_original_image = st.sidebar.checkbox('Show Original Image', False)

large_line_every = 5 # frequency of bold lines for easier orientation
                     # TODO: Possibly change to user input?

if show_original_image:
    st.image(image, caption='Uploaded Image')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

if blur_kernel_size: # TODO: kernel size must be odd (or figure out how gaussian_kernel_size can add padding)
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
    image = cv2.Gaussianblur_kernel_size(image, (blur_kernel_size, blur_kernel_size), 0)


h,w = image.shape
height_size = int(width_size * h / w)

w_ratio = w // width_size
h_ratio = h // height_size

downscale_image = cv2.resize(image, (height_size, width_size))  # Replace (800, 600) with your desired resolution
d_h, d_w = downscale_image.shape

image = cv2.resize(downscale_image, (height_size * h_ratio, width_size * w_ratio), interpolation=cv2.INTER_NEAREST)
_, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

# Add grid to the image based on the downscaled image
# TODO: 
if invert:
    image = cv2.bitwise_not(image)
    downscale_image = cv2.bitwise_not(downscale_image)

grid_color = (90, 60, 90)  # Define the color of the grid (in BGR format)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Draw grid 
# TODO: Lots of magical constants here, probably a better way of doing this
large_line_color = (100, 20, 20)
small_line_color = (80, 80, 80)
if show_grid:
    # Calculate the size of each grid cell
    cell_width = image.shape[1] // downscale_image.shape[1]
    cell_height = image.shape[0] // downscale_image.shape[0]

    # Draw vertical grid lines
    for x in range(0, downscale_image.shape[1]+1):
        if x % 5:
            line_width = 1
            line_color = small_line_color
        else:
            line_width = 3
            line_color = large_line_color
        x = x * cell_width
        cv2.line(image, (x, 0), (x, image.shape[0]), line_color, line_width)

    # Draw horizontal grid lines
    for y in range(0, downscale_image.shape[0]+1):
        y = y * cell_height
        if y % 5:
            line_width = 1
            line_color = small_line_color
        else:
            line_width = 3
            line_color = large_line_color
        cv2.line(image, (0, y), (image.shape[1], y), line_color, line_width)

# Display the binary image
st.image(image, caption='')
st.write(f'vyska: {downscale_image.shape[0]}')
st.write(f'sirka: {downscale_image.shape[1]}')

# Produce output for crocheting 
gray_image = (downscale_image > 128).astype(bool) # TODO: This is hacky, should be better way to do this
# TODO: Figure out what is good output strategy
if st.sidebar.button('Download Image'):
    cv2.imwrite('output_image.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    pdf = fpdf.FPDF()
    pdf.add_page()
    pdf.image('output_image.jpg', x=10, y=10, w=100, h=100)
    pdf.output('output.pdf', 'F')
