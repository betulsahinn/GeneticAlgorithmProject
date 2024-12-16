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


# Data loading and preprocessing
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
            for img_name in os.listdir(class_dir):
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


# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to manageable size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset (update path according to your setup)
dataset = Fruits360Dataset(root_dir='path/to/fruits-360/Training', transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


# Convert data to format suitable for MLPClassifier
def prepare_data(dataset):
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(dataloader))
    return images.reshape(len(dataset), -1).numpy(), labels.numpy()


X_train, y_train = prepare_data(train_dataset)
X_test, y_test = prepare_data(test_dataset)

# Genetic Algorithm Parameters
POPULATION_SIZE = 50
MUTATION_PROB = 0.05
CROSSOVER_PROB = 0.3
NUM_GENERATIONS = 1000

# Hyperparameter space (updated for Fruits-360 dataset)
HYPERPARAMETER_SPACE = {
    "learning_rate": (0.0001, 0.1),
    "batch_size": [16, 32, 64, 128, 256],
    "hidden_layers": (1, 5),
    "neurons_per_layer": (16, 128),
    "activation": ["relu", "tanh", "logistic"],
    "optimizer": ["sgd", "adam", "lbfgs"]
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
            "optimizer": random.choice(HYPERPARAMETER_SPACE["optimizer"])
        }
        population.append(individual)
    return population


def evaluate_individual(individual):
    model = MLPClassifier(
        hidden_layer_sizes=(individual["neurons_per_layer"],) * individual["hidden_layers"],
        activation=individual["activation"],
        solver=individual["optimizer"],
        learning_rate_init=individual["learning_rate"],
        max_iter=200,
        random_state=42
    )
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    except Exception as e:
        print(f"Error training model: {e}")
        return 0.0


def select_parents(population, fitness_scores):
    # Tournament selection
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
        if gene_to_mutate == "learning_rate":
            individual[gene_to_mutate] = np.random.uniform(*HYPERPARAMETER_SPACE[gene_to_mutate])
        elif gene_to_mutate in ["hidden_layers", "neurons_per_layer"]:
            individual[gene_to_mutate] = random.randint(*HYPERPARAMETER_SPACE[gene_to_mutate])
        else:
            individual[gene_to_mutate] = random.choice(HYPERPARAMETER_SPACE[gene_to_mutate])
    return individual


def genetic_algorithm():
    population = initialize_population()
    best_individual = None
    best_fitness = 0

    for generation in range(NUM_GENERATIONS):
        fitness_scores = [evaluate_individual(ind) for ind in population]
        current_best = max(fitness_scores)
        if current_best > best_fitness:
            best_fitness = current_best
            best_individual = population[fitness_scores.index(current_best)]

        print(f"Generation {generation + 1}, Best Fitness: {best_fitness:.4f}")

        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = select_parents(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population[:POPULATION_SIZE]

    return best_individual, best_fitness


# Run the genetic algorithm
if __name__ == "__main__":
    print("Starting genetic algorithm optimization...")
    best_solution, best_accuracy = genetic_algorithm()
    print("\nOptimization completed!")
    print(f"Best hyperparameters found: {best_solution}")
    print(f"Best accuracy achieved: {best_accuracy:.4f}")