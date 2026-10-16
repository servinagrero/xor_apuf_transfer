import numpy.typing as npt
import torch
import torch.nn as nn
from tqdm import tqdm


class FiLMLayer(nn.Module):
    """Device-specific conditioning."""

    def __init__(self, device_emb_dim: int, feature_dim: int):
        super().__init__()
        self.film_gen = nn.Sequential(
            nn.Linear(device_emb_dim, 64), nn.GELU(), nn.Linear(64, feature_dim * 2)
        )

    def forward(self, features, dev_emb):
        film_params = self.film_gen(dev_emb)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        return features * (1.0 + gamma) + beta


class PUFNetwork(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        num_devices: int,
        device_emb_dim: int,
        hidden_dim: int = 128,
        k_xor: int = 1,
    ):
        super().__init__()
        self.num_devices = num_devices
        self.k_xor = k_xor
        self.embedding = nn.Embedding(num_devices, device_emb_dim)

        # Layer 1: Models the k independent delay lines
        self.fc1 = nn.Linear(n_inputs, hidden_dim)

        # Layer 2: Models the XOR logic combination
        if self.k_xor > 1:
            self.fc1_b = nn.Linear(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()
        self.film = FiLMLayer(device_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, d_idx):
        all_dev_ids = torch.arange(self.num_devices, device=x.device)
        all_embs = self.embedding(all_dev_ids)
        all_film_params = self.film.film_gen(all_embs)
        all_gamma, all_beta = torch.chunk(all_film_params, 2, dim=-1)

        gamma = all_gamma[d_idx]
        beta = all_beta[d_idx]

        # 1. Evaluate the nominal delay lines
        h = self.fc1(x)

        # 2. Apply FiLM HERE: Adapt the physical delays to the target board's profile
        h_modulated = self.relu(h * (1.0 + gamma) + beta)

        # 3. Combine the adapted delays using the XOR logic layer
        if self.k_xor > 1:
            h_out = self.fc1_b(h_modulated)
            h_out = self.relu(h_out)
            return self.fc2(h_out)
        else:
            return self.fc2(h_modulated)


class PUFModel:
    def __init__(
        self,
        n_inputs,
        num_devices: int,
        device_emb_dim: int,
        hidden_dim: int,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        k_xor: int = 1,
    ):
        self.n_devices = num_devices
        self.model = PUFNetwork(
            n_inputs, num_devices, device_emb_dim, hidden_dim, k_xor=k_xor
        )
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def save(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str):
        state_dict = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def fit(
        self,
        X,
        D,
        Y,
        X_val,
        D_val,
        Y_val,
        patience: int | None = None,
        show_progress: bool = False,
    ):
        # # Transform pypuf {-1, 1} to {0, 1} for BCELoss
        X_train_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        D_train_t = torch.tensor(D, dtype=torch.long, device=self.device)
        Y_train_t = torch.tensor(Y, dtype=torch.float32, device=self.device).view(-1, 1)

        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        D_val_t = torch.tensor(D_val, dtype=torch.long, device=self.device)
        Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=self.device).view(
            -1, 1
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        num_samples = X_train_t.size(0)

        best_val_loss = float("inf")
        trigger_times = 0

        # Use dynamic patience if provided, otherwise default to 10
        current_patience = patience if patience is not None else 10

        # Calculate validation less frequently (e.g., max 50 checks total)
        val_check_interval = max(1, self.epochs // 50)

        train_log = []
        self.model.train()

        if show_progress:
            pbar = tqdm(range(self.epochs), desc="Training")
        else:
            pbar = range(self.epochs)

        best_model_state = {}

        for epoch in pbar:
            total_loss = 0.0

            # BYPASS DATALOADER: Fast C++ level shuffling and slicing
            indices = torch.randperm(num_samples, device=self.device)

            steps = 0
            for start_idx in range(0, num_samples, self.batch_size):
                b_idx = indices[start_idx : start_idx + self.batch_size]
                bx = X_train_t[b_idx]
                bd = D_train_t[b_idx]
                by = Y_train_t[b_idx]

                optimizer.zero_grad(set_to_none=True)
                preds = self.model(bx, bd)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1

            if epoch % val_check_interval == 0 or epoch == self.epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(X_val_t, D_val_t)
                    val_loss = criterion(val_preds, Y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Only copy state dict when necessary
                    best_model_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                    trigger_times = 0
                else:
                    trigger_times += 1
                    if trigger_times >= current_patience:
                        self.model.load_state_dict(best_model_state)
                        break
                self.model.train()  # Back to train mode

            # Avoid division by zero if dataset is smaller than batch size
            avg_loss = total_loss / max(1, steps)
            if show_progress:
                pbar.set_postfix({"loss": f"{avg_loss:.3e}"})  # type: ignore

        if best_model_state:
            self.model.load_state_dict(best_model_state)
        return train_log

    def predict(self, X, D) -> npt.NDArray:
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        D = torch.tensor(D, dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(X, D)
            # OPTIMIZATION: Apply sigmoid here since we removed it from forward()
            preds = torch.sigmoid(logits)
        return preds.cpu().numpy()

    def set_transfer_mode(self, freeze: bool):
        """
        Freezes or unfreezes the shared backbone. During transfer, we only want the embedding to be trainable.

        self.model.embedding.weight remains trainable so we can optimize the new device's embedding.
        """
        # Freeze shared feature extractor, FiLM generator, and head
        for param in self.model.fc1.parameters():
            param.requires_grad = not freeze
        for param in self.model.film.film_gen.parameters():
            param.requires_grad = not freeze
        for param in self.model.fc2.parameters():
            param.requires_grad = not freeze
        if self.model.k_xor > 1:
            for param in self.model.fc1_b.parameters():
                param.requires_grad = not freeze

    def fit_transfer(self, X, target_device_id, Y, X_val, D_val, Y_val, patience=None):
        self.set_transfer_mode(freeze=True)
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device).view(-1, 1)
        D_t = torch.full(
            (len(X),), target_device_id, dtype=torch.long, device=self.device
        )
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=self.device).view(
            -1, 1
        )
        D_val_t = torch.tensor(D_val, dtype=torch.long, device=self.device)
        optimizer = torch.optim.LBFGS(
            filter(lambda parameter: parameter.requires_grad, self.model.parameters()),
            max_iter=1,
            lr=0.1,
            line_search_fn="strong_wolfe",
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        best_loss = float("inf")
        best_state = None
        stale = 0
        current_patience = patience if patience is not None else 10

        try:
            for _ in range(20):
                self.model.train()

                def closure():
                    optimizer.zero_grad()
                    loss = criterion(self.model(X_t, D_t), Y_t)
                    loss.backward()
                    return loss

                optimizer.step(closure)
                self.model.eval()
                with torch.no_grad():
                    validation_loss = criterion(
                        self.model(X_val_t, D_val_t), Y_val_t
                    ).item()
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_state = {
                        name: value.clone()
                        for name, value in self.model.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
                    if stale >= current_patience:
                        break
            if best_state is not None:
                self.model.load_state_dict(best_state)
        finally:
            self.set_transfer_mode(freeze=False)

    def extract_film_parameters(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        self.model.eval()

        # Create a tensor of all device IDs [0, 1, ..., num_devices - 1]
        device_ids = torch.arange(self.n_devices, dtype=torch.long, device=self.device)

        with torch.no_grad():
            # Get the learned embedding vectors for all devices
            embeddings = self.model.embedding(device_ids)

            # Pass embeddings through the FiLM generator
            film_params = self.model.film.film_gen(embeddings)

            # Split into Gamma and Beta
            gamma, beta = torch.chunk(film_params, 2, dim=-1)

        return gamma.cpu().numpy(), beta.cpu().numpy(), embeddings.cpu().numpy()
