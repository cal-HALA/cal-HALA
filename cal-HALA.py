import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import time
import os
from sklearn.metrics import f1_score


HIDDEN_DIM = 128
LR = 0.01
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-8


TEACH_LOSS_WEIGHT = 6
CONTRASTIVE_LOSS_WEIGHT = 1
TEMPERATURE = 0.2


def get_acc(phi_ik, truthfile):
    phi_ik = phi_ik.detach().cpu().numpy()
    score = (phi_ik == phi_ik.max(axis=1, keepdims=True)).astype(float)
    score /= score.sum(axis=1, keepdims=True)
    return score[truthfile.item.values, truthfile.truth.values].sum() / truthfile.shape[0]


def get_macro_f1(phi_ik, truthfile):
    phi_ik = phi_ik.detach().cpu().numpy()
    y_pred = np.argmax(phi_ik, axis=1)
    y_pred_test = y_pred[truthfile.item.values]
    y_true = truthfile.truth.values
    return f1_score(y_true, y_pred_test, average='macro', zero_division=0)


def build_graph_with_agreement(item2workers, num_workers):
    intersection_counts = defaultdict(int)
    agreement_counts = defaultdict(int)

    for i, annos in item2workers.items():
        workers = [w for w, _ in annos]
        labels = [l for _, l in annos]

        if len(workers) < 2: continue

        for j in range(len(workers)):
            for k in range(j + 1, len(workers)):
                u, v = workers[j], workers[k]
                if u == v: continue

                pair = tuple(sorted((u, v)))
                intersection_counts[pair] += 1
                if labels[j] == labels[k]:
                    agreement_counts[pair] += 1

    if not intersection_counts:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float)

    src_list = []
    dst_list = []
    weight_list = []

    for (u, v), count in intersection_counts.items():
        if count < 1: continue

        if count < 4:
            weight = 0.0
        else:
            agree = agreement_counts[(u, v)]
            ratio = agree / count
            weight = (ratio - 0.5) * 2.0

        src_list.extend([u, v])
        dst_list.extend([v, u])
        weight_list.extend([weight, weight])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)

    return edge_index, edge_weight


def get_teacher_data(df, Z_mv, num_workers, num_classes):
    worker_counts = df['worker'].value_counts()
    top_k = int(len(worker_counts) * 0.3)
    if top_k < 1: top_k = 1

    teacher_ids = worker_counts.index[:top_k].values
    teacher_mask = torch.zeros(num_workers, dtype=torch.bool, device=DEVICE)
    teacher_mask[teacher_ids] = True

    mv_labels = torch.argmax(Z_mv, dim=1).cpu().numpy()
    item_mv_map = {item_id: label for item_id, label in enumerate(mv_labels)}

    counts = defaultdict(lambda: np.zeros((num_classes, num_classes)))
    df_teachers = df[df['worker'].isin(teacher_ids)]

    for _, row in df_teachers.iterrows():
        w = int(row['worker'])
        item = int(row['item'])
        label = int(row['label'])

        if item in item_mv_map:
            mv_truth = item_mv_map[item]
            counts[w][mv_truth][label] += 1

    target_cms = torch.zeros(num_workers, num_classes, num_classes, device=DEVICE)
    for w in teacher_ids:
        mat = torch.tensor(counts[w], dtype=torch.float32, device=DEVICE)
        row_sums = mat.sum(dim=1, keepdim=True)
        mat = torch.where(row_sums > 0, mat / row_sums, torch.eye(num_classes, device=DEVICE))
        target_cms[w] = mat

    return teacher_mask, target_cms


def compute_initial_features(df, Z_mv, num_workers, num_classes):
    mv_labels = torch.argmax(Z_mv, dim=1).cpu().numpy()
    temp_df = df.copy()
    temp_df['mv_truth'] = mv_labels[temp_df['item'].values]

    counts = temp_df.groupby(['worker', 'mv_truth', 'label']).size().reset_index(name='cnt')

    init_cms = torch.zeros(num_workers, num_classes, num_classes, device=DEVICE)
    w_idx = torch.tensor(counts['worker'].values, dtype=torch.long, device=DEVICE)
    t_idx = torch.tensor(counts['mv_truth'].values, dtype=torch.long, device=DEVICE)
    l_idx = torch.tensor(counts['label'].values, dtype=torch.long, device=DEVICE)
    val = torch.tensor(counts['cnt'].values, dtype=torch.float32, device=DEVICE)

    init_cms.index_put_((w_idx, t_idx, l_idx), val)

    row_sums = init_cms.sum(dim=2, keepdim=True)
    mask_has_data = row_sums > 0
    init_cms[mask_has_data.squeeze(-1)] = init_cms[mask_has_data.squeeze(-1)] / row_sums[mask_has_data.squeeze(-1)]
    identity = 0.2*torch.eye(num_classes, device=DEVICE).unsqueeze(0).expand(num_workers, -1, -1)

    mask_no_data = (row_sums == 0)
    init_cms = torch.where(mask_no_data, identity, init_cms)

    return init_cms.view(num_workers, -1)


class AnnotatorNet(nn.Module):
    def __init__(self, num_workers, num_classes, hidden_dim, init_features_tensor):
        super().__init__()
        self.num_classes = num_classes
        input_dim = num_classes * num_classes

        self.node_features = nn.Parameter(init_features_tensor.clone().detach())

        self.feature_proj = nn.Linear(input_dim, hidden_dim)

        self.W_gat = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.a_gat = nn.Linear(hidden_dim, 1, bias=False)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes * num_classes)
        )

    def forward(self, edge_index):
        h = self.feature_proj(self.node_features)  # [N, H]

        if edge_index.size(1) > 0:
            src, dst = edge_index

            h_cat = torch.cat([h[src], h[dst]], dim=1)

            te = self.W_gat(h_cat)  # [E, H]
            te = F.leaky_relu(te, 0.2)
            x = self.a_gat(te).squeeze(-1)  # [E]

            alpha = torch.exp(x)
            row_sum = torch.zeros(h.size(0), device=h.device)
            row_sum.index_add_(0, dst, alpha)
            alpha = alpha / (row_sum[dst] + EPS)

            msg = h[src] * alpha.unsqueeze(-1)
            aggr = torch.zeros_like(h)
            aggr.index_add_(0, dst, msg)

            h = h + 0.6*aggr

        cm_logits = self.decoder(h).view(-1, self.num_classes, self.num_classes)
        confusion_matrices = F.softmax(cm_logits, dim=2)

        return confusion_matrices, h


def info_nce_loss(embeddings, edge_index, edge_weight, temperature=0.1):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    num_nodes = embeddings.shape[0]
    device = embeddings.device

    pos_mask = edge_weight >= 0.5
    neg_mask = edge_weight <= -0.4

    if pos_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    neg_exp_sum_per_node = torch.zeros(num_nodes, device=device)
    if neg_mask.sum() > 0:
        neg_u = edge_index[0][neg_mask]
        neg_v = edge_index[1][neg_mask]
        sim_neg = (embeddings[neg_u] * embeddings[neg_v]).sum(dim=1) / temperature
        exp_sim_neg = torch.exp(sim_neg)
        neg_exp_sum_per_node.index_add_(0, neg_u, exp_sim_neg)

    pos_u = edge_index[0][pos_mask]
    pos_v = edge_index[1][pos_mask]
    sim_pos = (embeddings[pos_u] * embeddings[pos_v]).sum(dim=1) / temperature
    exp_sim_pos = torch.exp(sim_pos)

    current_neg_sum = neg_exp_sum_per_node[pos_u]
    denominator = exp_sim_pos + current_neg_sum
    loss = torch.log(denominator) - sim_pos

    return loss.mean()


def calHALA(dataset):
    path = f"./datasets/{dataset}/"
    label_file = os.path.join(path, "label.csv")
    if not os.path.exists(label_file): return None
    df = pd.read_csv(label_file)
    df.columns = df.columns.str.strip().str.lower()

    num_workers = df["worker"].max() + 1
    num_items = df["item"].max() + 1
    num_classes = df["label"].max() + 1

    worker2items = defaultdict(list)
    item2workers = defaultdict(list)
    labels_tensor = torch.tensor(df['label'].values, dtype=torch.long, device=DEVICE)
    workers_tensor = torch.tensor(df['worker'].values, dtype=torch.long, device=DEVICE)
    items_tensor = torch.tensor(df['item'].values, dtype=torch.long, device=DEVICE)

    for _, r in df.iterrows():
        i, j, y = int(r["item"]), int(r["worker"]), int(r["label"])
        item2workers[i].append((j, y))
        worker2items[j].append((i, y))

    Z = torch.zeros(num_items, num_classes, device=DEVICE)
    for i, annos in item2workers.items():
        for _, y in annos:
            Z[i, y] += 1
    Z /= (Z.sum(axis=1, keepdims=True) + EPS)
    temp = Z.sum(axis=0)
    temp /= temp.sum()

    teacher_mask, teacher_target_cms = get_teacher_data(df, Z, num_workers, num_classes)

    edge_index, edge_weight = build_graph_with_agreement(item2workers, num_workers)
    edge_index = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)

    init_features = compute_initial_features(df, Z, num_workers, num_classes)

    model = AnnotatorNet(num_workers, num_classes, HIDDEN_DIM, init_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    flag = 1

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        cms, embeddings = model(edge_index)

        obs_cms = cms[workers_tensor]
        target_labels = labels_tensor.view(-1, 1, 1).expand(-1, num_classes, 1)
        prob_y_given_k = obs_cms.gather(2, target_labels).squeeze(2)
        log_prob = torch.log(prob_y_given_k + EPS)
        item_log_prob = torch.zeros(num_items, num_classes, device=DEVICE)
        item_log_prob.index_add_(0, items_tensor, log_prob)
        loss_nll = -torch.sum(Z.detach() * item_log_prob) / num_items

        eye = torch.eye(num_classes, device=DEVICE).unsqueeze(0)
        loss_reg = torch.mean((cms - eye) ** 2)

        if teacher_mask.sum() > 0:
            loss_teach = F.mse_loss(cms[teacher_mask], teacher_target_cms[teacher_mask])
        else:
            loss_teach = torch.tensor(0.0, device=DEVICE)

        loss_contrastive = info_nce_loss(embeddings, edge_index, edge_weight, temperature=TEMPERATURE)

        total_loss = (loss_nll +
                      0.5*loss_reg +
                      TEACH_LOSS_WEIGHT * loss_teach +
                      CONTRASTIVE_LOSS_WEIGHT * loss_contrastive)

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            cms, _ = model(edge_index)
            obs_cms = cms[workers_tensor]
            target_labels = labels_tensor.view(-1, 1, 1).expand(-1, num_classes, 1)
            prob_y_given_k = obs_cms.gather(2, target_labels).squeeze(2)
            log_prob = torch.log(prob_y_given_k + EPS)
            log_posterior = torch.zeros(num_items, num_classes, device=DEVICE)
            log_posterior.index_add_(0, items_tensor, log_prob)

            if epoch < EPOCHS * 0.2:
                log_pi = torch.log(temp + EPS)
                SMOOTH = 1
            elif flag == 1:
                log_pi = torch.log(temp + EPS)

                SMOOTH = 0.00001*num_items
                if SMOOTH > 1:
                    SMOOTH = 1

                flag = 0

            log_posterior = log_posterior + SMOOTH*log_pi.unsqueeze(0)
            Z = F.softmax(log_posterior, dim=1)


    return Z


datasets = ['LabelMe', 'aircrowd6', 'valence5', 'valence7', 'CF', 'MS',
            's4_Dog_data', 's4_Face_Sentiment_Identification', 's5_AdultContent', 'web']


if __name__ == "__main__":
    iteration = 10

    iter_accs = [[] for _ in range(iteration)]
    iter_f1s = [[] for _ in range(iteration)]
    iter_times = [[] for _ in range(iteration)]

    for dataset in datasets:
        print(dataset)

        dataset_accs = []
        dataset_f1s = []
        dataset_times = []

        for i in range(iteration):
            truth_file = './datasets/' + dataset + '/truth.csv'
            datafile = "./datasets/" + dataset + "/label.csv"
            starttime = time.time()
            df_label = pd.read_csv(datafile)

            df_truth = pd.read_csv(truth_file)

            phi_ik = calHALA(dataset)

            duration = time.time() - starttime
            accuracy = get_acc(phi_ik, df_truth)
            macro_f1 = get_macro_f1(phi_ik, df_truth)

            dataset_accs.append(accuracy)
            dataset_f1s.append(macro_f1)
            dataset_times.append(duration)

            iter_accs[i].append(accuracy)
            iter_f1s[i].append(macro_f1)
            iter_times[i].append(duration)

            print(f"Iteration {i + 1}: Acc={accuracy:.4f}, Macro-F1={macro_f1:.4f}, Time={duration:.4f}s")

        dataset_acc_mean = np.mean(dataset_accs)
        dataset_acc_std = np.std(dataset_accs)
        dataset_f1_mean = np.mean(dataset_f1s)
        dataset_f1_std = np.std(dataset_f1s)
        dataset_time_mean = np.mean(dataset_times)
        dataset_time_std = np.std(dataset_times)

        print(f"\n--- {dataset} ---")
        print(f"Accuracy: {dataset_acc_mean:.4f} ± {dataset_acc_std:.4f}")
        print(f"Macro-F1: {dataset_f1_mean:.4f} ± {dataset_f1_std:.4f}")
        print(f"Time: {dataset_time_mean:.4f} ± {dataset_time_std:.4f}s\n")

    avg_accs = [np.mean(iter_accs[i]) for i in range(iteration)]
    avg_f1s = [np.mean(iter_f1s[i]) for i in range(iteration)]
    avg_times = [np.mean(iter_times[i]) for i in range(iteration)]

    overall_acc_mean = np.mean(avg_accs)
    overall_acc_std = np.std(avg_accs)
    overall_f1_mean = np.mean(avg_f1s)
    overall_f1_std = np.std(avg_f1s)
    overall_time_mean = np.mean(avg_times)
    overall_time_std = np.std(avg_times)

    print("=" * 60)
    print(f"Overall Accuracy: {overall_acc_mean:.4f} ± {overall_acc_std:.4f}")
    print(f"Overall Macro-F1: {overall_f1_mean:.4f} ± {overall_f1_std:.4f}")
    print(f"Overall Time: {overall_time_mean:.4f} ± {overall_time_std:.4f}s")
    print("=" * 60)

