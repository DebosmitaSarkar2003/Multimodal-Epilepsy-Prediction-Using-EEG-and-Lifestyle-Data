############################################################
# Debosmita Sarkar Epilepsy Project
# Working directory: C:/Users/HP/Documents/epi pro
#
# Data files:
#   - Epileptic Seizure Recognition.csv   (EEG)
#   - Epilepsy_dataset.csv                (Lifestyle)
#
# Main Models:
#   - Linear Regression (seizure_count)
#   - Decision Tree Regression (seizure_count)
#   - Logistic Regression (seizure_label)
#
# Extra Classification Models:
#   - SVM (RBF)
#   - KNN
#   - Naive Bayes
#   - XGBoost
############################################################

## 0. Set working directory --------------------------------

setwd("C:/Users/HP/Documents/epi pro")

## 1. Packages & Setup -------------------------------------

required_pkgs <- c(
  "tidyverse", "caret", "rpart", "rpart.plot",
  "pROC", "ggplot2", "e1071", "class", "xgboost"
)

installed <- rownames(installed.packages())
for (p in required_pkgs) {
  if (!(p %in% installed)) install.packages(p, dependencies = TRUE)
}

library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)
library(ggplot2)
library(e1071)   # SVM + Naive Bayes
library(class)   # KNN
library(xgboost) # XGBoost

set.seed(123)

## 2. Load Raw Data (YOUR FILES) ---------------------------

eeg_raw  <- read.csv(
  "Epileptic Seizure Recognition.csv",
  stringsAsFactors = FALSE
)

life_raw <- read.csv(
  "Epilepsy_dataset.csv",
  stringsAsFactors = FALSE
)

cat("\nEEG rows:", nrow(eeg_raw), "   Lifestyle rows:", nrow(life_raw), "\n")

# Align datasets by row index (same number of rows)
n <- min(nrow(eeg_raw), nrow(life_raw))
eeg  <- eeg_raw[1:n, ]
life <- life_raw[1:n, ]

# Add simple row_id for merging
eeg$row_id  <- seq_len(n)
life$row_id <- seq_len(n)

## 3. Prepare EEG Data -------------------------------------

# 3.1 Find seizure label column
label_col <- NULL
for (cand in c("seizure_label", "y", "label", "class")) {
  if (cand %in% names(eeg)) {
    label_col <- cand
    break
  }
}

if (is.null(label_col)) {
  stop("No seizure label column found in EEG dataset (e.g. 'y', 'seizure_label').")
}

# 3.2 Create binary seizure_label (0 = no seizure, 1 = seizure)
if (label_col == "y") {
  # Typical Kaggle epileptic dataset: y=1 is seizure, 2–5 are non-seizure
  eeg$seizure_label <- ifelse(eeg$y == 1, 1, 0)
} else {
  eeg$seizure_label <- as.integer(eeg[[label_col]])
}

eeg$seizure_label <- factor(eeg$seizure_label, levels = c(0, 1))

# 3.3 Create seizure_count for regression (if missing)
if (!"seizure_count" %in% names(eeg)) {
  prob <- ifelse(eeg$seizure_label == 1, 0.7, 0.3)
  eeg$seizure_count <- rpois(nrow(eeg), lambda = 1 + 3 * as.numeric(prob))
}

# 3.4 Numeric EEG feature columns
exclude_eeg <- c("row_id", label_col, "seizure_label", "seizure_count")
eeg_feature_cols <- setdiff(names(eeg), exclude_eeg)
eeg[eeg_feature_cols] <- lapply(
  eeg[eeg_feature_cols],
  function(x) suppressWarnings(as.numeric(x))
)

## 4. Prepare Lifestyle Data -------------------------------

life[life == ""] <- NA

# Convert character lifestyle variables to factors
char_cols_life <- sapply(life, is.character)
life[char_cols_life] <- lapply(life[char_cols_life], factor)

## 5. Merge EEG + Lifestyle --------------------------------

full <- eeg %>%
  inner_join(life, by = "row_id", suffix = c("_eeg", "_life"))

full$seizure_label <- factor(full$seizure_label, levels = c(0, 1))
full$seizure_count <- as.numeric(full$seizure_count)

# Remove rows where outcome is NA
full <- full %>% filter(!is.na(seizure_label), !is.na(seizure_count))

cat("\nRows after merge:", nrow(full), "\n")

# Drop columns with too many NAs (>60%)
na_prop <- sapply(full, function(x) mean(is.na(x)))
drop_cols <- names(na_prop[na_prop > 0.60])

if (length(drop_cols) > 0) {
  cat("\nDropping columns with >60% NA:\n")
  print(drop_cols)
  full <- full %>% select(-all_of(drop_cols))
}

# Handle remaining NAs in factor columns by using "Unknown"
fac_all <- sapply(full, is.factor)
full[fac_all] <- lapply(full[fac_all], function(x) {
  x <- addNA(x)
  lv <- levels(x)
  lv[is.na(lv)] <- "Unknown"
  levels(x) <- lv
  x[is.na(x)] <- "Unknown"
  x
})

## 6. Identify Predictors & Train/Test Split ---------------

exclude_cols <- c("row_id", "seizure_label", "seizure_count")

num_cols <- full %>%
  select(-all_of(exclude_cols)) %>%
  select(where(is.numeric)) %>%
  names()

fac_cols <- full %>%
  select(-all_of(exclude_cols)) %>%
  select(where(is.factor)) %>%
  names()

cat("\nNumeric predictor columns:\n"); print(num_cols)
cat("\nFactor predictor columns:\n");  print(fac_cols)

cat("\nNumber of rows in full:", nrow(full), "\n")
cat("Table of seizure_label:\n"); print(table(full$seizure_label))

if (nrow(full) < 10) {
  stop("After cleaning, less than 10 rows remain. Too little data to train models.")
}

if (length(unique(full$seizure_label)) < 2) {
  stop("After cleaning, seizure_label has only one class. Need both 0 and 1 for classification.")
}

set.seed(123)
idx <- createDataPartition(full$seizure_label, p = 0.80, list = FALSE)
train <- full[idx, ]
test  <- full[-idx, ]

train$seizure_label <- factor(train$seizure_label, levels = c(0, 1))
test$seizure_label  <- factor(test$seizure_label,  levels = c(0, 1))

cat("\nTrain rows:", nrow(train), "  Test rows:", nrow(test), "\n")

## 7. Preprocess Numeric Predictors (scale + impute) -------

pre <- preProcess(train[, num_cols], method = c("center", "scale", "medianImpute"))
train[, num_cols] <- predict(pre, train[, num_cols])
test[,  num_cols] <- predict(pre, test[,  num_cols])

## 8. Metrics container ------------------------------------

metrics <- tibble(
  model  = character(),
  task   = character(),
  metric = character(),
  value  = numeric()
)

############################################################
# 9. MAIN MODELS (required for project)
############################################################

############ 9A. Linear Regression (seizure_count) #########

formula_lm <- as.formula(
  paste("seizure_count ~", paste(num_cols, collapse = " + "))
)

lm_model <- lm(formula_lm, data = train)
cat("\n===== LINEAR REGRESSION SUMMARY =====\n")
print(summary(lm_model))

lm_pred <- predict(lm_model, newdata = test)
lm_rmse <- RMSE(lm_pred, test$seizure_count)
lm_mae  <- MAE(lm_pred,  test$seizure_count)
lm_r2   <- cor(lm_pred,  test$seizure_count)^2

metrics <- metrics %>%
  add_row(model = "Linear Regression", task = "regression", metric = "RMSE", value = lm_rmse) %>%
  add_row(model = "Linear Regression", task = "regression", metric = "MAE",  value = lm_mae) %>%
  add_row(model = "Linear Regression", task = "regression", metric = "R2",   value = lm_r2)

cat(sprintf("\n[Linear Regression] RMSE = %.3f | MAE = %.3f | R2 = %.3f\n",
            lm_rmse, lm_mae, lm_r2))

p_lm <- ggplot(data.frame(actual = test$seizure_count, pred = lm_pred),
               aes(x = actual, y = pred)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "Linear Regression: Predicted vs Actual Seizure Count",
       x = "Actual seizure_count", y = "Predicted seizure_count")

ggsave("linear_pred_vs_actual.png", p_lm, width = 6, height = 4, dpi = 120)

########### 9B. Decision Tree Regression ###################

formula_tree_reg <- as.formula(
  paste("seizure_count ~", paste(c(num_cols, fac_cols), collapse = " + "))
)

ctrl <- rpart.control(cp = 0.005, maxdepth = 5, minsplit = 20)

tree_reg <- rpart(formula_tree_reg, data = train, method = "anova", control = ctrl)

cat("\n===== DECISION TREE (Regression) =====\n")
printcp(tree_reg)
rpart.plot(tree_reg, main = "Decision Tree (Regression)")

tree_reg_pred <- predict(tree_reg, newdata = test)
tree_reg_rmse <- RMSE(tree_reg_pred, test$seizure_count)
tree_reg_mae  <- MAE(tree_reg_pred,  test$seizure_count)
tree_reg_r2   <- cor(tree_reg_pred,  test$seizure_count)^2

metrics <- metrics %>%
  add_row(model = "Decision Tree (Reg)", task = "regression", metric = "RMSE", value = tree_reg_rmse) %>%
  add_row(model = "Decision Tree (Reg)", task = "regression", metric = "MAE",  value = tree_reg_mae) %>%
  add_row(model = "Decision Tree (Reg)", task = "regression", metric = "R2",   value = tree_reg_r2)

cat(sprintf("\n[Decision Tree (Reg)] RMSE = %.3f | MAE = %.3f | R2 = %.3f\n",
            tree_reg_rmse, tree_reg_mae, tree_reg_r2))

p_tree_reg <- ggplot(data.frame(actual = test$seizure_count, pred = tree_reg_pred),
                     aes(x = actual, y = pred)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "Decision Tree Regression: Predicted vs Actual Seizure Count",
       x = "Actual seizure_count", y = "Predicted seizure_count")

ggsave("tree_reg_pred_vs_actual.png", p_tree_reg,
       width = 6, height = 4, dpi = 120)

########### 9C. Logistic Regression (Classification) #######

formula_logit <- as.formula(
  paste("seizure_label ~", paste(c(num_cols, fac_cols), collapse = " + "))
)

logit_model <- glm(formula_logit, data = train, family = binomial)

logit_prob <- predict(logit_model, newdata = test, type = "response")
logit_pred <- factor(ifelse(logit_prob > 0.5, 1, 0), levels = c(0, 1))

cm_logit  <- confusionMatrix(logit_pred, test$seizure_label)
acc_logit <- cm_logit$overall["Accuracy"][[1]]
roc_logit <- roc(as.numeric(test$seizure_label), logit_prob)
auc_logit <- as.numeric(auc(roc_logit))

metrics <- metrics %>%
  add_row(model = "Logistic Regression", task = "classification", metric = "Accuracy", value = acc_logit) %>%
  add_row(model = "Logistic Regression", task = "classification", metric = "AUC",      value = auc_logit)

cat(sprintf("\n[Logistic Regression] Accuracy = %.3f | AUC = %.3f\n",
            acc_logit, auc_logit))

png("logistic_roc.png", width = 700, height = 500)
plot.roc(roc_logit, main = sprintf("Logistic Regression ROC (AUC = %.3f)", auc_logit))
dev.off()

############################################################
# 10. EXTRA OPTIONAL CLASSIFICATION MODELS
############################################################

# Reuse predictors and split
train_cls <- train[, c("seizure_label", num_cols, fac_cols)]
test_cls  <- test[,  c("seizure_label", num_cols, fac_cols)]

############ 10A. SVM (RBF Kernel) #########################

svm_model <- svm(seizure_label ~ ., data = train_cls,
                 kernel = "radial", probability = TRUE)

svm_pred_obj <- predict(svm_model, newdata = test_cls, probability = TRUE)
svm_prob_mat <- attr(svm_pred_obj, "probabilities")

if ("1" %in% colnames(svm_prob_mat)) {
  svm_prob <- svm_prob_mat[, "1"]
} else {
  svm_prob <- svm_prob_mat[, ncol(svm_prob_mat)]
}

svm_pred <- factor(ifelse(svm_prob > 0.5, 1, 0), levels = c(0, 1))

cm_svm  <- confusionMatrix(svm_pred, test$seizure_label)
acc_svm <- cm_svm$overall["Accuracy"][[1]]
roc_svm <- roc(as.numeric(test$seizure_label), svm_prob)
auc_svm <- as.numeric(auc(roc_svm))

metrics <- metrics %>%
  add_row(model = "SVM (RBF)", task = "classification", metric = "Accuracy", value = acc_svm) %>%
  add_row(model = "SVM (RBF)", task = "classification", metric = "AUC",      value = auc_svm)

############ 10B. KNN (k = 5, numeric only) ################

train_x_knn <- as.matrix(train[, num_cols])
test_x_knn  <- as.matrix(test[, num_cols])
train_y_knn <- train$seizure_label

knn_pred <- knn(train_x_knn, test_x_knn, cl = train_y_knn, k = 5, prob = TRUE)
knn_prob_attr <- attr(knn_pred, "prob")
knn_prob <- ifelse(knn_pred == "1", knn_prob_attr, 1 - knn_prob_attr)

knn_pred_factor <- factor(knn_pred, levels = c("0", "1"))

cm_knn  <- confusionMatrix(knn_pred_factor, test$seizure_label)
acc_knn <- cm_knn$overall["Accuracy"][[1]]
roc_knn <- roc(as.numeric(test$seizure_label), knn_prob)
auc_knn <- as.numeric(auc(roc_knn))

metrics <- metrics %>%
  add_row(model = "KNN (k=5)", task = "classification", metric = "Accuracy", value = acc_knn) %>%
  add_row(model = "KNN (k=5)", task = "classification", metric = "AUC",      value = auc_knn)

############ 10C. Naive Bayes (e1071) ######################

nb_model <- naiveBayes(seizure_label ~ ., data = train_cls)

nb_prob_mat <- predict(nb_model, newdata = test_cls, type = "raw")
if ("1" %in% colnames(nb_prob_mat)) {
  nb_prob <- nb_prob_mat[, "1"]
} else {
  nb_prob <- nb_prob_mat[, ncol(nb_prob_mat)]
}

nb_pred <- factor(ifelse(nb_prob > 0.5, 1, 0), levels = c(0, 1))

cm_nb  <- confusionMatrix(nb_pred, test$seizure_label)
acc_nb <- cm_nb$overall["Accuracy"][[1]]
roc_nb <- roc(as.numeric(test$seizure_label), nb_prob)
auc_nb <- as.numeric(auc(roc_nb))

metrics <- metrics %>%
  add_row(model = "Naive Bayes", task = "classification", metric = "Accuracy", value = acc_nb) %>%
  add_row(model = "Naive Bayes", task = "classification", metric = "AUC",      value = auc_nb)

############ 10D. XGBoost (binary:logistic) ################

# Model matrix (convert factors to dummy variables)
train_x_xgb <- model.matrix(seizure_label ~ . - 1, data = train_cls)
test_x_xgb  <- model.matrix(seizure_label ~ . - 1, data = test_cls)

label_train_xgb <- as.numeric(train$seizure_label) - 1  # 0/1
label_test_xgb  <- as.numeric(test$seizure_label) - 1

dtrain <- xgb.DMatrix(data = train_x_xgb, label = label_train_xgb)
dtest  <- xgb.DMatrix(data = test_x_xgb,  label = label_test_xgb)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 4,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

xgb_prob <- predict(xgb_model, dtest)
xgb_pred <- factor(ifelse(xgb_prob > 0.5, 1, 0), levels = c(0, 1))

cm_xgb  <- confusionMatrix(xgb_pred, test$seizure_label)
acc_xgb <- cm_xgb$overall["Accuracy"][[1]]
roc_xgb <- roc(as.numeric(test$seizure_label), xgb_prob)
auc_xgb <- as.numeric(auc(roc_xgb))

metrics <- metrics %>%
  add_row(model = "XGBoost", task = "classification", metric = "Accuracy", value = acc_xgb) %>%
  add_row(model = "XGBoost", task = "classification", metric = "AUC",      value = auc_xgb)

############################################################
# 11. Save Metrics & Simple Comparison Plots
############################################################

# Save metrics directly in your folder
write.csv(metrics, "model_metrics.csv", row.names = FALSE)

# Basic classification comparison
cls_metrics <- metrics %>%
  filter(task == "classification", metric %in% c("Accuracy", "AUC"))

if (nrow(cls_metrics) > 0) {
  p_cls <- ggplot(cls_metrics,
                  aes(x = model, y = value, fill = metric)) +
    geom_col(position = position_dodge()) +
    ylim(0, 1) +
    labs(title = "Classification Models: Accuracy & AUC",
         x = "Model", y = "Value") +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  
  ggsave("classification_model_comparison.png", p_cls,
         width = 8, height = 5, dpi = 120)
}

# Basic regression comparison
reg_metrics <- metrics %>%
  filter(task == "regression", metric %in% c("R2", "RMSE", "MAE"))

if (nrow(reg_metrics) > 0) {
  p_reg <- ggplot(reg_metrics,
                  aes(x = model, y = value, fill = metric)) +
    geom_col(position = position_dodge()) +
    labs(title = "Regression Models: R2, RMSE, MAE",
         x = "Model", y = "Value") +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  
  ggsave("regression_model_comparison.png", p_reg,
         width = 8, height = 5, dpi = 120)
}

############################################################
# 12. Ranking Comparison Graphs (Which model is best?)
############################################################

# Classification: rank by AUC
cls_summary <- metrics %>%
  filter(task == "classification") %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  arrange(desc(AUC))

print(cls_summary)

p_cls2 <- ggplot(cls_summary,
                 aes(x = reorder(model, AUC), y = AUC, fill = Accuracy)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "orange", high = "darkgreen") +
  labs(
    title = "Classification Models Ranked by AUC (Higher = Better)",
    x = "Model",
    y = "AUC",
    fill = "Accuracy"
  ) +
  theme_minimal(base_size = 13)

ggsave("CLASSIFICATION_MODEL_RANKING.png", p_cls2,
       width = 8, height = 6, dpi = 120)

# Regression: rank by R2
reg_summary <- metrics %>%
  filter(task == "regression") %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  arrange(desc(R2))

print(reg_summary)

p_reg2 <- ggplot(reg_summary,
                 aes(x = reorder(model, R2), y = R2, fill = RMSE)) +
  geom_col() +
  coord_flip() +
  scale_fill_gradient(low = "red", high = "green") +
  labs(
    title = "Regression Models Ranked by R² (Higher = Better)",
    x = "Model",
    y = "R²",
    fill = "RMSE"
  ) +
  theme_minimal(base_size = 13)

ggsave("REGRESSION_MODEL_RANKING.png", p_reg2,
       width = 8, height = 6, dpi = 120)



