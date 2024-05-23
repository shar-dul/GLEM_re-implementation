import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    # for each sample in dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]

def preprocess_deberta(features, tokenizer):
    max_length = 512
    input_ids_list = []
    attention_masks_list = []
    for row in features:
        dense_row = row.toarray().flatten()
        row_string = ' '.join(str(int(val)) for val in dense_row)
        inputs = tokenizer(row_string, max_length = max_length, padding = "max_length", truncation = True, return_tensors = "pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
    input_ids_tensor = torch.cat(input_ids_list, dim=0)
    attention_masks_tensor = torch.cat(attention_masks_list, dim=0)
    return input_ids_tensor, attention_masks_tensor

def predict(model, dataloader, device):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch_input_ids, batch_attention_mask, _ = batch
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            predictions = outputs.logits

            _, predicted = torch.max(predictions, 1)
            one_hot_predictions = torch.zeros_like(predictions)
            one_hot_predictions.scatter_(1, predicted.view(-1, 1), 1)
            all_predictions.append(one_hot_predictions.cpu().numpy())
    all_predictions = np.concatenate(all_predictions, axis=0)
    return all_predictions


def train_deberta(model, input_ids_tensor, attention_masks_tensor, train_mask, validation_mask, test_mask, labels, correct_labels, device):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # #device = torch.device("cpu")
    # model.to(device)

    labels = np.argmax(labels, axis=1)
    correct_labels_iter = np.argmax(correct_labels, axis=1)

    train_indices = train_mask.nonzero()[0]
    validation_indices = validation_mask.nonzero()[0]
    test_indices = test_mask.nonzero()[0]
    learning_rate = 0.0001
    batch_size = 8
    alpha = 0.6

    num_labeled = int((1-alpha) * len(train_indices))
    labeled_indices = train_indices[:num_labeled]
    unlabeled_indices = train_indices[num_labeled:]
    
    all_dataset = CustomDataset(input_ids_tensor, attention_masks_tensor, labels)
    labeled_dataset = CustomDataset(input_ids_tensor[labeled_indices], attention_masks_tensor[labeled_indices], correct_labels[labeled_indices])
    unlabeled_dataset = CustomDataset(input_ids_tensor[unlabeled_indices], attention_masks_tensor[unlabeled_indices], labels[unlabeled_indices])
    validation_dataset = CustomDataset(input_ids_tensor[validation_indices], attention_masks_tensor[validation_indices], correct_labels_iter[validation_indices])
    test_dataset = CustomDataset(input_ids_tensor[test_indices], attention_masks_tensor[test_indices], correct_labels_iter[test_indices])
    
    all_dataloader = DataLoader(all_dataset, batch_size=16, shuffle=False, num_workers=4)
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 3

    for epoch in range(num_epochs):
        print("Epoch:", epoch + 1)
        model.train()
        total_loss, iter = 0.0, 0
        for batch in labeled_dataloader:
            print("  Labeled Batch:", iter + 1)
            iter += 1
            batch_input_ids, batch_attention_mask, batch_labels = batch
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()

            # Forward pass for labeled data
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            predictions = outputs.logits
            labeled_loss = loss_function(predictions, batch_labels)
            
            labeled_loss.backward()
            optimizer.step()

            total_loss += labeled_loss.item()
            del batch_input_ids, batch_attention_mask, batch_labels, labeled_loss, outputs, predictions
            torch.cuda.empty_cache()

        for batch in unlabeled_dataloader:
            print("  Unlabeled Batch:", iter + 1)
            iter += 1
            batch_input_ids, batch_attention_mask, _ = batch  # labels are ignored for unlabeled data
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            optimizer.zero_grad()

            # Forward pass for unlabeled data
            with torch.no_grad():
                outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            pseudo_labels = torch.argmax(outputs.logits, dim=1).detach()
            pseudo_labels = pseudo_labels.to(device)
            
            # Calculate loss for pseudo-labeled data
            unlabeled_outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            unlabeled_loss = loss_function(unlabeled_outputs.logits, pseudo_labels)
            
            unlabeled_loss.backward()
            optimizer.step()

            total_loss += unlabeled_loss.item()
            del batch_input_ids, batch_attention_mask, pseudo_labels, unlabeled_loss, outputs, unlabeled_outputs
            torch.cuda.empty_cache()

        avg_loss = total_loss / (len(labeled_dataloader) + len(unlabeled_dataloader))
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}')

        # Validation
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            correct = 0
            total = 0
            for val_batch in validation_dataloader:
                val_input_ids, val_attention_mask, val_labels = val_batch
                val_input_ids = val_input_ids.to(device)
                val_attention_mask = val_attention_mask.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask)
                val_predictions = val_outputs.logits
                val_loss = loss_function(val_predictions, val_labels)

                total_val_loss += val_loss.item()

                _, predicted = torch.max(val_predictions, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

            avg_val_loss = total_val_loss / len(validation_dataloader)
            accuracy = correct / total
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')

    #Test the model
    model.eval()
    with torch.no_grad():
        total_test_loss = 0.0
        correct = 0
        total = 0
        for test_batch in test_dataloader:
            test_input_ids, test_attention_mask, test_labels = test_batch
            test_input_ids = test_input_ids.to(device)
            test_attention_mask = test_attention_mask.to(device)
            test_labels = test_labels.to(device)

            test_outputs = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
            test_predictions = test_outputs.logits
            test_loss = loss_function(test_predictions, test_labels)

            total_test_loss += test_loss.item()

            _, predicted = torch.max(test_predictions, 1)
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()

        avg_test_loss = total_test_loss / len(test_dataloader)
        accuracy = correct / total
        print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}')
    return model, predict(model, all_dataloader, device)