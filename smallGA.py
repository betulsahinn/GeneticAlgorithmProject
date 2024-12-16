import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pandas as pd
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import shutil
from pathlib import Path


def create_small_dataset(source_dir, target_dir, n_classes=5, n_images_per_class=50):
    """
    Kaynak dizinden daha küçük bir dataset oluşturur
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)

    classes = [d for d in sorted(os.listdir(source_dir))
               if os.path.isdir(os.path.join(source_dir, d)) and not d.startswith('.')][:n_classes]

    for class_name in classes:
        source_class_dir = os.path.join(source_dir, class_name)
        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir)

        valid_extensions = {'.jpg', '.jpeg', '.png'}
        images = [f for f in os.listdir(source_class_dir)
                  if os.path.isfile(os.path.join(source_class_dir, f)) and
                  not f.startswith('.') and
                  any(f.lower().endswith(ext) for ext in valid_extensions)]

        selected_images = images[:n_images_per_class]

        if len(selected_images) < n_images_per_class:
            print(f"Uyarı: {class_name} sınıfında istenen sayıda resim bulunamadı. "
                  f"Bulunan: {len(selected_images)}, İstenen: {n_images_per_class}")

        for img_name in selected_images:
            source_path = os.path.join(source_class_dir, img_name)
            target_path = os.path.join(target_class_dir, img_name)
            shutil.copy2(source_path, target_path)

    print(f"Küçük dataset oluşturuldu:")
    print(f"- {len(classes)} sınıf")
    for class_name in classes:
        n_images = len([f for f in os.listdir(os.path.join(target_dir, class_name))
                        if not f.startswith('.')])
        print(f"- {class_name}: {n_images} resim")


class Fruits360Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):  # Sadece dizinleri işle
                for img_name in os.listdir(class_dir):
                    if not img_name.startswith('.'):  # Gizli dosyaları atla
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resim boyutunu artırdık
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def prepare_data(source_dir, n_classes=5, n_images_per_class=50):
    """
    Küçük bir test dataseti hazırlar ve yükler
    """
    temp_dir = "temp_small_dataset"
    create_small_dataset(source_dir, temp_dir, n_classes, n_images_per_class)

    dataset = Fruits360Dataset(root_dir=temp_dir, transform=transform)

    # Test set oranını artırdık
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    def prepare_subset(subset):
        dataloader = DataLoader(subset, batch_size=len(subset), shuffle=False)
        images, labels = next(iter(dataloader))
        return images.reshape(len(subset), -1).numpy(), labels.numpy()

    X_train, y_train = prepare_subset(train_dataset)
    X_test, y_test = prepare_subset(test_dataset)

    return X_train, X_test, y_train, y_test


# Genetik Algoritma Parametreleri
POPULATION_SIZE = 30  # Popülasyon boyutunu artırdık
MUTATION_PROB = 0.1  # Mutasyon olasılığını artırdık
CROSSOVER_PROB = 0.7  # Crossover olasılığını artırdık
NUM_GENERATIONS = 20  # Nesil sayısını artırdık

# Hiperparametre uzayı
HYPERPARAMETER_SPACE = {
    "learning_rate": (0.0001, 0.01),  # Learning rate aralığını daralttık
    "batch_size": [32, 64, 128],  # Batch size seçeneklerini düzenledik
    "hidden_layers": (1, 3),  # Max hidden layer sayısını azalttık
    "neurons_per_layer": (32, 128),  # Minimum nöron sayısını artırdık
    "activation": ["relu", "tanh"],  # Activation fonksiyonlarını azalttık
    "optimizer": ["adam", "sgd"],  # Optimizer seçeneklerini azalttık
    "alpha": (0.0001, 0.01)  # L2 regularization için alpha parametresi ekledik
}


def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = {
            "learning_rate": np.random.uniform(*HYPERPARAMETER_SPACE["learning_rate"]),
            "batch_size": random.choice(HYPERPARAMETER_SPACE["batch_size"]),
            "hidden_layers": random.randint(*HYPERPARAMETER_SPACE["hidden_layers"]),
            "neurons_per_layer": random.randint(*HYPERPARAMETER_SPACE["neurons_per_layer"]),
            "activation": random.choice(HYPERPARAMETER_SPACE["activation"]),
            "optimizer": random.choice(HYPERPARAMETER_SPACE["optimizer"]),
            "alpha": np.random.uniform(*HYPERPARAMETER_SPACE["alpha"])  # Yeni alpha parametresi
        }
        population.append(individual)
    return population


def evaluate_individual(individual, X_train, X_test, y_train, y_test):
    model = MLPClassifier(
        hidden_layer_sizes=(individual["neurons_per_layer"],) * individual["hidden_layers"],
        activation=individual["activation"],
        solver=individual["optimizer"],
        learning_rate_init=individual["learning_rate"],
        max_iter=200,  # Maximum iterasyon sayısını artırdık
        alpha=individual["alpha"],  # L2 regularization
        batch_size=individual["batch_size"],
        random_state=42
    )
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    except Exception as e:
        print(f"Model eğitim hatası: {e}")
        return 0.0


def select_parents(population, fitness_scores):
    tournament_size = 5
    tournament_winners = []
    for _ in range(2):
        tournament = random.sample(list(enumerate(population)), tournament_size)
        winner = max(tournament, key=lambda x: fitness_scores[x[0]])
        tournament_winners.append(winner[1])
    return tournament_winners[0], tournament_winners[1]


def crossover(parent1, parent2):
    if random.random() < CROSSOVER_PROB:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = {**parent1, **{k: parent2[k] for k in list(parent2.keys())[crossover_point:]}}
        child2 = {**parent2, **{k: parent1[k] for k in list(parent1.keys())[crossover_point:]}}
        return child1, child2
    return parent1, parent2


def mutate(individual):
    if random.random() < MUTATION_PROB:
        gene_to_mutate = random.choice(list(individual.keys()))
        if gene_to_mutate in ["learning_rate", "alpha"]:
            individual[gene_to_mutate] = np.random.uniform(*HYPERPARAMETER_SPACE[gene_to_mutate])
        elif gene_to_mutate in ["hidden_layers", "neurons_per_layer"]:
            individual[gene_to_mutate] = random.randint(*HYPERPARAMETER_SPACE[gene_to_mutate])
        else:
            individual[gene_to_mutate] = random.choice(HYPERPARAMETER_SPACE[gene_to_mutate])
    return individual


def genetic_algorithm(X_train, X_test, y_train, y_test):
    population = initialize_population()
    best_individual = None
    best_fitness = 0

    for generation in range(NUM_GENERATIONS):
        fitness_scores = [evaluate_individual(ind, X_train, X_test, y_train, y_test) for ind in population]
        current_best = max(fitness_scores)
        if current_best > best_fitness:
            best_fitness = current_best
            best_individual = population[fitness_scores.index(current_best)]

        print(f"Nesil {generation + 1}, En İyi Uygunluk: {best_fitness:.4f}")

        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population[:POPULATION_SIZE]

    return best_individual, best_fitness


if __name__ == "__main__":
    SOURCE_DIR = "/Users/betulsahin/PycharmProjects/GeneticAlgorithmProject/dataset/archive/fruits-360_dataset_100x100/fruits-360/Training"

    print(f"Dataset yolu: {SOURCE_DIR}")

    if not os.path.exists(SOURCE_DIR):
        print(f"Hata: Dataset dizini bulunamadı: {SOURCE_DIR}")
        exit(1)

    print("Küçük test dataseti hazırlanıyor...")

    try:
        X_train, X_test, y_train, y_test = prepare_data(
            SOURCE_DIR,
            n_classes=5,
            n_images_per_class=50  # Her sınıf için resim sayısını artırdık
        )

        print("Genetik algoritma optimizasyonu başlatılıyor...")
        best_solution, best_accuracy = genetic_algorithm(X_train, X_test, y_train, y_test)

        print("\nOptimizasyon tamamlandı!")
        print(f"En iyi hiperparametreler: {best_solution}")
        print(f"En iyi doğruluk oranı: {best_accuracy:.4f}")

    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        if os.path.exists("temp_small_dataset"):
            shutil.rmtree("temp_small_dataset")