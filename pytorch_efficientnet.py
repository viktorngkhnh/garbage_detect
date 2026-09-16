import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# --- Ustawienia Globalne ---
# random_seed: Używany do inicjalizacji generatorów liczb losowych, aby zapewnić powtarzalność wyników.
# Dzięki temu, przy każdym uruchomieniu skryptu z tym samym seedem, operacje losowe (np. podział danych, inicjalizacja wag) będą przebiegać tak samo.
random_seed = 42
# batch_size: Liczba próbek przetwarzanych jednocześnie podczas jednej iteracji treningu lub ewaluacji.
# Większy batch_size może przyspieszyć trening, ale wymaga więcej pamięci GPU.
batch_size = 32
# device: Określa, czy obliczenia będą wykonywane na GPU (cuda) czy CPU.
# Ten skrypt zakłada dostępność CUDA i ustawia 'cuda' na stałe.
device = torch.device('cuda')

# Ustawienie seeda dla PyTorch dla wszystkich urządzeń (CPU i CUDA) w celu zapewnienia reprodukowalności.
torch.manual_seed(random_seed)

# --- Definicja Modelu ---
# Definiujemy klasę EfficientNet dziedziczącą po nn.Module, co jest standardem w PyTorch do tworzenia modeli.
class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        # Wywołanie konstruktora klasy bazowej (nn.Module).
        super().__init__()
        # Użyj predefiniowanego modelu EfficientNet-B0 z wagami wytrenowanymi na zbiorze ImageNet.
        # models.EfficientNet_B0_Weights.IMAGENET1K_V1 zapewnia użycie konkretnej, stabilnej wersji wag.
        # Transfer learning: wykorzystujemy wiedzę zdobytą przez model na dużym zbiorze danych (ImageNet)
        # do naszego specyficznego zadania klasyfikacji śmieci.
        # EfficientNet to nowoczesna architektura zaprojektowana do osiągnięcia najlepszego kompromisu
        # między dokładnością a efektywnością obliczeniową poprzez skalowanie głębokości, szerokości i rozdzielczości.
        self.network = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # EfficientNet-B0 oryginalnie klasyfikuje do 1000 klas ImageNet.
        # Musimy zastąpić ostatnią warstwę klasyfikującą (classifier)
        # nową warstwą, która będzie miała tyle neuronów wyjściowych, ile mamy klas śmieci.
        # num_ftrs: Pobieramy liczbę cech wejściowych do oryginalnej warstwy classifier.
        # W EfficientNet classifier to moduł składający się z Dropout i Linear layer.
        num_ftrs = self.network.classifier[1].in_features
        # self.network.classifier: Zastępujemy cały classifier nową warstwą liniową (nn.Linear).
        # Ta nowa warstwa przyjmuje num_ftrs cech na wejściu i wyprowadza num_classes wartości (logity).
        self.network.classifier = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, xb):
        # Metoda forward definiuje, jak dane przechodzą przez model.
        return self.network(xb)

# --- Funkcje Treningu i Ewaluacji ---
# Dekorator @torch.no_grad() wyłącza obliczanie gradientów dla tej funkcji.
# Jest to kluczowe podczas ewaluacji, ponieważ przyspiesza obliczenia i zmniejsza zużycie pamięci,
# a gradienty nie są potrzebne, gdy nie aktualizujemy wag modelu.
@torch.no_grad()
def evaluate(model, val_loader):
    # model.eval(): Przełącza model w tryb ewaluacji.
    # Ma to znaczenie dla warstw takich jak Dropout czy BatchNorm, które zachowują się inaczej podczas treningu i ewaluacji.
    model.eval()
    # Listy do przechowywania strat i dokładności dla każdej paczki (batch) danych.
    batch_losses = []
    batch_accs = []
    
    # Iterujemy po danych w walidacyjnym DataLoaderze.
    for images, labels in val_loader:
        # Przenosimy obrazy i etykiety na wybrane urządzenie (GPU).
        images, labels = images.to(device), labels.to(device)
        
        # Przepuszczamy obrazy przez model, aby uzyskać predykcje (logity).
        out = model(images)
        # Obliczamy stratę (loss) używając funkcji cross-entropy.
        # Cross-entropy jest standardową funkcją straty dla problemów klasyfikacji wieloklasowej.
        loss = F.cross_entropy(out, labels)
        
        # Obliczanie dokładności (accuracy).
        # torch.max(out, dim=1) zwraca (wartości maksymalne, indeksy maksymalnych wartości) dla każdego wiersza (dim=1).
        # Indeksy te odpowiadają przewidzianym klasom.
        _, preds = torch.max(out, dim=1)
        # Porównujemy przewidziane klasy (preds) z prawdziwymi etykietami (labels).
        # torch.sum(...) zlicza, ile razy predykcje były poprawne.
        # .item() konwertuje tensor jedoelementowy do liczby Pythonowej.
        # Dzielimy przez liczbę próbek w batchu, aby uzyskać średnią dokładność dla batcha.
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        
        # Dodajemy stratę i dokładność do list.
        # .detach() jest używane na stracie, aby odłączyć ją od grafu obliczeniowego,
        # co zapobiega niepotrzebnemu śledzeniu gradientów i oszczędza pamięć.
        batch_losses.append(loss.detach())
        batch_accs.append(acc)
            
    # Obliczamy średnią stratę i dokładność dla całej epoki walidacyjnej.
    # torch.stack(...) łączy listę tensorów w jeden tensor.
    # .mean() oblicza średnią.
    # .item() konwertuje wynikowy tensor do liczby Pythonowej.
    epoch_loss = torch.stack(batch_losses).mean().item()
    epoch_acc = torch.stack(batch_accs).mean().item()
    return {'val_loss': epoch_loss, 'val_acc': epoch_acc}

# Główna funkcja treningowa.
# epochs: liczba epok treningu.
# lr: współczynnik uczenia (learning rate).
# model: model do trenowania.
# train_loader: DataLoader dla danych treningowych.
# val_loader: DataLoader dla danych walidacyjnych.
# opt_func: funkcja optymalizatora (domyślnie Adam).
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    # Lista do przechowywania historii metryk (strata, dokładność) dla każdej epoki.
    history = []
    # Inicjalizacja optymalizatora.
    # model.parameters() dostarcza optymalizatorowi parametry modelu, które mają być aktualizowane.
    optimizer = opt_func(model.parameters(), lr)
    
    # Pętla główna treningu, iterująca przez zadaną liczbę epok.
    for epoch in range(epochs):
        # model.train(): Przełącza model w tryb treningu.
        # Ważne dla warstw jak Dropout i BatchNorm.
        model.train()
        # Lista do przechowywania strat treningowych dla każdej paczki w danej epoce.
        train_losses = []
        
        # Używamy tqdm do wyświetlania paska postępu dla pętli treningowej.
        # desc: opis wyświetlany obok paska postępu.
        train_progress = tqdm(train_loader, desc=f'Epoka {epoch+1}/{epochs} - Trening')
        # Pętla wewnętrzna, iterująca po paczkach danych z treningowego DataLoadera.
        for images, labels in train_progress:
            # Przenosimy dane na urządzenie (GPU).
            images, labels = images.to(device), labels.to(device)
            
            # --- Krok 1: Forward pass --- 
            # Przepuszczamy obrazy przez model, aby uzyskać predykcje.
            out = model(images)
            
            # --- Krok 2: Obliczenie straty --- 
            # Obliczamy stratę między predykcjami a prawdziwymi etykietami.
            loss = F.cross_entropy(out, labels)
            # Zapisujemy stratę. .detach().cpu() kopiuje tensor straty na CPU i odłącza od grafu,
            # co jest dobrą praktyką, aby nie zapełniać pamięci GPU historią strat.
            train_losses.append(loss.detach().cpu())
            
            # --- Krok 3: Backward pass (propagacja wsteczna) --- 
            # Obliczamy gradienty straty względem parametrów modelu.
            loss.backward()
            
            # --- Krok 4: Aktualizacja wag --- 
            # Optymalizator aktualizuje wagi modelu na podstawie obliczonych gradientów.
            optimizer.step()
            
            # --- Krok 5: Wyzerowanie gradientów --- 
            # Gradienty są akumulowane, więc trzeba je wyzerować po każdej aktualizacji wag,
            # aby gradienty z poprzedniej paczki nie wpływały na obliczenia dla bieżącej.
            optimizer.zero_grad()
            
            # Aktualizacja paska postępu tqdm o bieżącą wartość straty.
            train_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # --- Faza walidacji po każdej epoce --- 
        # Wywołujemy funkcję evaluate, aby uzyskać metryki na zbiorze walidacyjnym.
        result = evaluate(model, val_loader)
        # Obliczamy średnią stratę treningową dla zakończonej epoki.
        result['train_loss'] = torch.stack(train_losses).mean().item()
        
        # Wyświetlamy metryki dla bieżącej epoki.
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))
        # Dodajemy wyniki z epoki do historii.
        history.append(result)
        
    # Zwracamy historię treningu.
    return history

# --- Funkcja do Wykresów ---
def plot_results(history):
    plt.figure(figsize=(12, 4))
    
    # Wykres dokładności
    plt.subplot(1, 2, 1)
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność Walidacyjna')
    plt.title('Dokładność vs. Liczba Epok (EfficientNet-B0)')
    plt.grid(True)
    
    # Wykres straty
    plt.subplot(1, 2, 2)
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx', label='Treningowa')
    plt.plot(val_losses, '-rx', label='Walidacyjna')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
    plt.title('Strata vs. Liczba Epok (EfficientNet-B0)')
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('./models/pytorch_efficientnet', exist_ok=True)
    plt.savefig('./models/pytorch_efficientnet/pytorch_efficientnet_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# --- Główna część skryptu ---
if __name__ == "__main__":
    print("=== ROZPOCZĘCIE TRENINGU EfficientNet-B0 (Uproszczony, CUDA-Only, Split 60/10/30) ===")
    print(f"Używane urządzenie: {device}")

    data_dir = "data/garbage_classification"
    if not os.path.isdir(data_dir):
        print(f"Błąd: Katalog danych '{data_dir}' nie został znaleziony.")
        exit()
        
    # Transformacje dla EfficientNet-B0 (rozmiar obrazu 224x224)
    # EfficientNet-B0 został trenowany na obrazach 224x224, więc używamy tego rozmiaru
    # zamiast 256x256 jak w ResNet50 dla lepszej kompatybilności
    transformations = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()
    ])
    
    print("Ładowanie datasetu...")
    try:
        dataset = ImageFolder(data_dir, transform=transformations)
    except FileNotFoundError:
        print(f"Błąd: Nie można załadować datasetu ze ścieżki '{data_dir}'. Sprawdź ścieżkę.")
        exit()
        
    print(f"Znalezione klasy: {dataset.classes}")
    print(f"Całkowita liczba próbek: {len(dataset)}")
    
    total_size = len(dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    print(f"Podział danych - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)
    
    model = EfficientNet(len(dataset.classes)).to(device)
    
    print("Ewaluacja przed treningiem:")
    initial_result = evaluate(model, val_dl) 
    print(f"Początkowa val_loss: {initial_result['val_loss']:.4f}, val_acc: {initial_result['val_acc']:.4f}")
    
    num_epochs = 8
    opt_func = torch.optim.Adam
    # Dla EfficientNet-B0 używamy nieco wyższego learning rate
    # ponieważ model ma inną architekturę i może wymagać różnych hiperparametrów
    lr = 1e-4
    
    print(f"\nParametry treningu:")
    print(f"- Model: EfficientNet-B0")
    print(f"- Epoki: {num_epochs}")
    print(f"- Optimizer: Adam")
    print(f"- Learning rate: {lr}")
    print(f"- Batch size: {batch_size} (train), {batch_size*2} (val)")
    print(f"- Rozmiar obrazu: 224x224 (optymalne dla EfficientNet-B0)\n")
    
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    
    results = {
        'approach': 'PyTorch EfficientNet-B0 (CUDA-Only, Split 60/10/30)',
        'model': 'EfficientNet-B0',
        'epochs': num_epochs,
        'learning_rate': lr,
        'batch_size': batch_size,
        'optimizer': 'Adam',
        'image_size': '224x224',
        'final_val_acc': history[-1]['val_acc'],
        'final_val_loss': history[-1]['val_loss'],
        'final_train_loss': history[-1]['train_loss'],
        'history': history,
        'data_split': {'train': len(train_ds), 'val': len(val_ds), 'test': len(test_ds)}
    }
    
    os.makedirs('./models/pytorch_efficientnet', exist_ok=True)
    results_path = './models/pytorch_efficientnet/pytorch_efficientnet_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_results(history)
    
    print(f"\n=== WYNIKI KOŃCOWE EfficientNet-B0 ===")
    print(f"Końcowa dokładność walidacyjna: {history[-1]['val_acc']:.4f} ({history[-1]['val_acc']*100:.2f}%)")
    print(f"Końcowa strata walidacyjna: {history[-1]['val_loss']:.4f}")
    print(f"Końcowa strata treningowa: {history[-1]['train_loss']:.4f}")
    
    model_path = './models/pytorch_efficientnet/pytorch_efficientnet_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel zapisany jako '{model_path}'")
    print(f"Wyniki zapisane w '{results_path}'")
    print(f"Wykres zapisany w './models/pytorch_efficientnet/pytorch_efficientnet_results.png'")
