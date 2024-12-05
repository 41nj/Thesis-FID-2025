import matplotlib.pyplot as plt
import json
import numpy as np

def balkendiagramm():
    with open('mix_logfile.json', 'r') as file:
        data = json.load(file)

    x_labels = []
    y_values = []

    for entry in data:  
        class1 = entry["results"]["class1"].replace(".n.01", "")
        class2 = entry["results"]["class2:"].replace(".n.01", "")
        fid_score = entry["results"]["FID_Score"]
        
        x_labels.append(f"{class2}")
        y_values.append(fid_score)

    print(x_labels)
    print(y_values)

    plt.figure(figsize=(10, 6))
    plt.bar(x_labels, y_values, color='skyblue')

    plt.title("FID Scores for Class Comparisons: Soup")
    plt.xlabel("Class")
    plt.ylabel("FID Score")
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("soup_scores_plot.png")
    plt.show()

def scatter():
    with open('mix_logfile.json', 'r') as file:
        data = json.load(file)

    x_labels = []
    y_labels = []
    fid_scores = []

    for entry in data: 
        class1 = entry["results"]["class1"].replace(".n.01", "")
        class2 = entry["results"]["class2:"].replace(".n.01", "")
        fid_score = entry["results"]["FID_Score"]
        
        x_labels.append(class1)
        y_labels.append(class2)
        fid_scores.append(fid_score)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_labels, y_labels, c=fid_scores, cmap='viridis', s=100)

    plt.colorbar(scatter, label="FID Score")

    plt.title("FID Scores for Class Comparisons")
    plt.xlabel("Class 1")
    plt.ylabel("Class 2")
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("mix_scores_scatterplot.png")  
    plt.show()

def heatmap():

    with open('noise_mix_logfile.json', 'r') as file:
        data = json.load(file)

    class1_list = []
    class2_list = []
    fid_scores = []

    for entry in data:
        class1 = entry["results"]["class1"].replace(".n.01", "")
        class2 = entry["results"]["class2:"].replace(".n.01", "")
        fid_score = entry["results"]["FID_Score"]
        class1_list.append(class1)
        class2_list.append(class2)
        fid_scores.append(fid_score)

    unique_class1 = sorted(set(class1_list))
    unique_class2 = sorted(set(class2_list))

    heatmap_matrix = np.full((len(unique_class2), len(unique_class1)), np.nan)

    for c1, c2, score in zip(class1_list, class2_list, fid_scores):
        i = unique_class2.index(c2)
        j = unique_class1.index(c1)
        heatmap_matrix[i, j] = score

    for i in range(len(unique_class2)):
        for j in range(len(unique_class1)):
            if np.isnan(heatmap_matrix[i, j]) and not np.isnan(heatmap_matrix[j, i]):
                heatmap_matrix[i, j] = heatmap_matrix[j, i]
            elif np.isnan(heatmap_matrix[j, i]) and not np.isnan(heatmap_matrix[i, j]):
                heatmap_matrix[j, i] = heatmap_matrix[i, j]

    for i in range(len(unique_class2)):
        for j in range(len(unique_class1)):
            if i < j:  
                heatmap_matrix[i, j] = np.nan

    print(heatmap_matrix)

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_matrix, cmap="autumn", aspect="auto")

    plt.xticks(range(len(unique_class1)), unique_class1, rotation=45, ha='right')
    plt.yticks(range(len(unique_class2)), unique_class2)
    plt.colorbar(label="FID Score")

    plt.title("FID Scores Heatmap for Class Comparisons")


    plt.tight_layout()
    plt.savefig("noise_fid_scores_heatmap.png")
    plt.show()

def extract_fid_scores_from_json(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        fid_scores = {}
        
        for entry in data:
            class1 = entry["results"]["class1"].replace(".n.01", "")
            class2 = entry["results"]["class2:"].replace(".n.01", "")
            
            sorted_pair = tuple(sorted([class1, class2]))
            
            fid_score = entry['results']['FID_Score']
            
            fid_scores[sorted_pair] = fid_score
        
        return fid_scores

#LINIENDIA VGL
def vgl_linien():
    
    fid_scores_file1 = extract_fid_scores_from_json("json_files/mix_logfile.json")
    fid_scores_file2 = extract_fid_scores_from_json("json_files/768_mix_logfile.json")
    fid_scores_file3 = extract_fid_scores_from_json("json_files/192_mix_logfile.json")
    fid_scores_file4 = extract_fid_scores_from_json("json_files/64_mix_logfile.json")

    class_pairs = list(set(fid_scores_file1.keys()).union(set(fid_scores_file2.keys())))

    class_pair_strings = [f"{pair[0]} vs {pair[1]}" for pair in class_pairs]

    file1_scores = [fid_scores_file1.get(pair, None) for pair in class_pairs]
    file2_scores = [fid_scores_file2.get(pair, None) for pair in class_pairs]
    file3_scores = [fid_scores_file3.get(pair, None) for pair in class_pairs]
    file4_scores = [fid_scores_file4.get(pair, None) for pair in class_pairs]



    plt.figure(figsize=(10, 6))
    plt.plot(class_pair_strings, file1_scores, label="2048", marker='o', color='blue')  
    plt.plot(class_pair_strings, file2_scores, label="768", marker='o', color='green')   
    plt.plot(class_pair_strings, file3_scores, label="192", marker='o', color='red') 
    plt.plot(class_pair_strings, file4_scores, label="64", marker='o', color='orange')  
    

    plt.xlabel('Class Pair (Class1 vs Class2)')
    plt.ylabel('FID Score')
    plt.title('FID Score Comparison between Class Pairs')

    plt.xticks(rotation=45, ha="right")
    plt.legend()




    plt.tight_layout()
    plt.savefig("TEST.png")
    plt.show()



import matplotlib.pyplot as plt
import json

def liniendia3():
    # Dateien und Farben für Modelle
    json_files = [
        ('json_files/mix_logfile.json', '2048', 'lightblue'),
        ('json_files/768_mix_logfile.json', '768', 'g'),
        ('json_files/192_mix_logfile.json', '192', 'b'),
        ('json_files/64_mix_logfile.json', '64', 'r')
    ]

    plt.figure(figsize=(10, 6))
    
    for file_name, label, color in json_files:
        # Daten aus jeder JSON-Datei lesen
        with open(file_name, 'r') as file:
            data = json.load(file)
        
        noise_values = []
        fid_scores = []

        for entry in data:
            noise_values.append(entry['results']['noise'])
            fid_scores.append(entry['results']['FID_Score'])

        # Linie plotten
        plt.plot(noise_values, fid_scores, marker='o', linestyle='-', color=color, label=label)
        print(fid_scores)

        # Werte über den Linien anzeigen
        for x, y in zip(noise_values, fid_scores):
            if color == 'b':  # Blau: Werte oben rechts
                plt.text(x - 0.5, y + 3, f'{round(y, 1)}', fontsize=8, color=color)
            elif color == 'r':  # Rot: Werte unter den Punkten
                plt.text(x+0.5, y - 20, f'{round(y, 1)}', fontsize=8, ha='center', color=color)
            elif color == 'g':  # Grün: Werte unter den Punkten
                plt.text(x-1.5, y - 15, f'{round(y, 1)}', fontsize=8, ha='center', color=color)


    # Achsenbeschriftungen und Titel
    plt.xlabel('Noise Level')
    plt.ylabel('FID Score')
    plt.title('FID Score with Different Noise Levels')
    plt.legend(title="Models")

    # Legende hinzufügen
    plt.legend(title="Models")
    plt.tight_layout()
    plt.savefig("multi_model_plot.png")
    plt.show()


    
if __name__ == '__main__':
    vgl_linien()