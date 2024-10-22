import matplotlib.pyplot as plt
import test
import numpy as np
import os

def plot_fid_scores():

    """fid_scores_single, fid_scores_multiple= test.run_tests()
    
    x_labels = ['0/0', '0/1','0/2','0/3','0/4','0/5','0/6','0/7','0/8', '0/9']  

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fid_scores_multiple)), fid_scores_multiple, marker='o')

    # dia
    plt.xticks(range(len(fid_scores_multiple)), x_labels) 
    plt.xlabel('FID Scores')
    plt.ylabel('Score Value')
    plt.title('FID Scores for Different Tests')
    plt.grid()
    plt.tight_layout()

    # show diagram
    plt.show()"""

    fid_scores_single, fid_scores_multiple = test.run_tests()  # get FID scores from the test module
    
    # labels for x-axis
    #x_labels = [f'9/{i}' for i in range(len(fid_scores_single))]  
    x_labels = ['no cutout', 'cutout: 70%', 'cutout: 50%','cutout: 20%']
    # create plot
    plt.figure(figsize=(10, 5))
    plt.bar(x_labels, fid_scores_multiple, color='b', alpha=0.7)

    # plot settings
    plt.xlabel('Test Runs')  
    plt.ylabel('FID Score')  
    plt.title('FID Scores for cropped images')  
    plt.grid(axis='y')  
    plt.tight_layout()  

   
    plt.show()


