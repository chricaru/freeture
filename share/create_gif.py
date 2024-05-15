#
#
#
#
# In order to create frames and video, launch this tool like this.
#
#then launch ffmpeg like this:
#F:\ffmpeg-master-latest-win64-gpl\bin\ffmpeg -framerate 24 -i C:\Users\anovati\Desktop\ITTO09_20240510\captures\Frame%03d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4

import argparse
import glob
import os
from PIL import Image
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.visualization import (AsinhStretch, LinearStretch, ZScaleInterval, LogStretch)
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy.ndimage import gaussian_filter
from astropy.visualization import simple_norm

def linear_stretch(data, scale_min=None, scale_max=None):
    if scale_min is None:
        scale_min = np.min(data)
    if scale_max is None:
        scale_max = np.max(data)
    return 65535 * (data - scale_min) / (scale_max - scale_min)


def process_fits(data,show=False):
    # Normalize the data using a logarithmic stretch to enhance faint stars
    norm = ImageNormalize(data, stretch=LogStretch())

    # Applying a Gaussian filter for noise reduction
    data_smoothed = gaussian_filter(data, sigma=0.1)
    #data_smoothed = data
    # Again apply normalization to the smoothed data
    norm_smoothed = ImageNormalize(data_smoothed, stretch=LogStretch())

    # Plotting original and processed data
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        img_orig = ax[0].imshow(data, norm=norm, origin='lower', cmap='gray')
        img_proc = ax[1].imshow(data_smoothed, norm=norm_smoothed, origin='lower', cmap='gray')
        ax[0].set_title('Original Image')
        ax[1].set_title('Processed Image')
        plt.colorbar(img_orig, ax=ax[0])
        plt.colorbar(img_proc, ax=ax[1])
        plt.show()


    #norm  = simple_norm(data_smoothed, 'log', percent=99)  # Adjust 'percent' to clip outliers
    norm  = simple_norm(data_smoothed, 'linear', percent=99)  # Adjust 'percent' to clip outliers
    image_data = norm(data_smoothed) * 255
    image_data = np.clip(image_data, 0, 255).astype(np.uint8)

    return image_data  # Return or save as needed


def fits_to_image(fits_file, save_png=False, output_dir=None,counter=0):
    try:
        with fits.open(fits_file) as hdul:
            data = hdul[0].data.astype(np.float64)
            if data is None:
                raise ValueError("No data found in FITS file.")
            
            
            image_data  = process_fits(data)
            
            
            # Convert to image and save as PNG
            image = Image.fromarray(image_data , mode='L')
        
            if save_png:
                counter_str = f"{counter:03d}"  # zero-pad counter to 3 digits
                #image_filename = os.path.basename(fits_file).replace('.fit', '.png')
                image_filename = f"Frame{counter_str}.png"
                image_path = os.path.join(output_dir, image_filename)
                image.save(image_path, 'PNG')
                print(f"PNG saved at {image_path}")

            return image
    except (OSError, ValueError, TypeError) as e:
        print(f"Error processing {fits_file}: {e}")
        return None

def create_gif(directory, duration, save_tiff, counter=0):
    # Find all FITS files in the specified directory
    fits_files = sorted(glob.glob(os.path.join(directory, "*.fit")), key=lambda x: os.path.basename(x))
    images = []
    for fits_file in fits_files:
        image = fits_to_image(fits_file, save_tiff, directory, counter)
        counter += 1 
        if image is not None:
            images.append(image)
    
    if not images:
        print("No valid images to create GIF.")
        return

    # Set the output GIF file path
    gif_path = os.path.join(directory, "output.gif")
    images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=duration*1000, loop=0)
    print(f"GIF created at {gif_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a GIF from FITS files.')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing FITS files')
    parser.add_argument('-t', '--time', type=int, default=1, help='Frame duration in seconds')
    parser.add_argument('--save-png', action='store_true', help='Save each frame as a PNG file')
    
    args = parser.parse_args()
    create_gif(args.directory, args.time, args.save_png)
