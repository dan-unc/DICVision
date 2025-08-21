import os
import numpy as np
import muDIC as dic
from muDIC import vlab
import matplotlib.pyplot as plt

output_dir = "deformation_frames" #directory in which your frames go
if not(os.path.isdir(output_dir)):
    os.makedirs(output_dir)
    
# # Image Stack Generation

n_cycles = 1 #number of strain cycles?
frames_per_cycle = 16 #number of frames in one cyle
total_frames = n_cycles * frames_per_cycle #400 frames
amplitude = 2000 #amplitude of the sine wave (what are the units?)
time_values = np.linspace(0, 2 * np.pi * n_cycles,
                          total_frames, endpoint=False) #time axis

image_shape = (128, 128) #size of the image
speckle_image = vlab.rosta_speckle(image_shape,
                                   dot_size=4,
                                   density=0.32,
                                   smoothness=2.0) #as shown in the virtual experiment

for i, t in enumerate(time_values):
    strain_xx = amplitude * np.sin(t) #horizontal strain
    F_t = np.array([[1+strain_xx, 0.0],
                    [0.0, 1.0]], dtype=np.float64) #adding a 1 to avoid singularities
    
    image_deformer = vlab.imageDeformer_from_defGrad(F_t) #creation of deformer
    img_def = image_deformer(speckle_image) #application of deformation
    
    #save and display progress
    filename = os.path.join(output_dir, f"Aa_frame_{i:04d}.png")
    plt.imsave(filename, img_def[0], cmap='gray', vmin=0, vmax=1)
    
    print(f"Saved frame {i+1}/{total_frames}: {filename}")

# # Once we have the images, we simply carry out the strain calculation

path = r"deformation_frames"
image_stack = dic.image_stack_from_folder(path,file_type=".png")

mesher = dic.Mesher()
mesh = mesher.mesh(image_stack)

inputs = dic.DICInput(mesh,image_stack)
dic_job = dic.DICAnalysis(inputs)

results = dic_job.run()

fields = dic.Fields(results)

disp = fields.disp()

viz = dic.Visualizer(fields,images=image_stack)

viz.show(field="truestrain", component = (1,1), frame = total_frames-1)

