# ============================================================
# TEXT MINING + TOPIC-SENTIMENT CORRELATION (FIXED)
# ============================================================
# --- Load libraries ---
library(tidyverse)
library(tidytext)
library(tm)
library(wordcloud)
library(SnowballC)
library(ggplot2)
library(textdata)
library(topicmodels)
library(widyr)
library(ggraph)
library(igraph)
library(RColorBrewer)
library(plotly)
library(tidyr)
library(ggdendro)

# ------------------------------------------------------------
# Step 1: Load and Clean (Same as before)
# ------------------------------------------------------------
setwd("C:/R/physics1")
files <- list.files(pattern = "*.txt")
speeches <- map_df(files, function(f) {
  tibble(president = tools::file_path_sans_ext(f),
         text = paste(readLines(f, encoding = "UTF-8", warn = FALSE), collapse = " "))
})

# Load standard stop words
data("stop_words")

# Define custom stop words to remove non-conceptual terms
custom_stop_words <- tibble(
  word = c("graphene", "theory", "figure", "nobel", "physics"),
  lexicon = "custom"
)

tidy_speeches <- speeches %>%
  unnest_tokens(word, text) %>%
  mutate(word = str_replace_all(word, "[^a-zA-Z]", "")) %>%
  filter(word != "", nchar(word) > 2) %>%
  anti_join(stop_words, by = "word") %>%      # Remove standard words
  anti_join(custom_stop_words, by = "word")   # Remove your specific list

# ------------------------------------------------------------
# Step 2: TOPIC MODELING (LDA)
# ------------------------------------------------------------

# 1. Create Document Term Matrix
dtm <- tidy_speeches %>%
  count(president, word) %>%
  cast_dtm(president, word, n)

# 2. Run Latent Dirichlet Allocation (LDA)
# We use k = 4 because you have 4 speeches loaded
lda_model <- LDA(dtm, k = 4, control = list(seed =1200))

# 3. Extract Topic-Word probabilities (Beta)
top_terms <- tidy(lda_model, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  rename(term = term)

# 4. Extract Document-Topic probabilities (Gamma)
gamma_tidy <- tidy(lda_model, matrix = "gamma")

# 5. Load NRC Sentiment Lexicon
# Note: You may be prompted to download this in the console
nrc <- get_sentiments("nrc")

# 6. Create tidy_sent for the heatmap
tidy_sent <- tidy_speeches %>%
  inner_join(nrc, by = "word")

# ------------------------------------------------------------
# Step 3: Topic-Sentiment Correlation (GUARANTEED 4 BARS)
# ------------------------------------------------------------
# 1. Get word-topic assignments from the LDA model for ALL words
# This provides a much larger dataset than 'top_terms'
all_topic_words <- tidy(lda_model, matrix = "beta")

topic_sent <- all_topic_words %>%
  inner_join(nrc, by = c("term" = "word")) %>%
  # Sum the beta (probability) to find the strongest sentiments per topic
  group_by(topic, sentiment) %>%
  summarise(beta_sum = sum(beta), .groups = "drop") %>%
  # 2. Force exactly 4 sentiments per topic
  group_by(topic) %>%
  slice_max(beta_sum, n = 4, with_ties = FALSE) %>%
  ungroup()

p_topic_sent <- ggplot(topic_sent,
                       aes(x = reorder_within(sentiment, beta_sum, topic),
                           y = beta_sum,
                           fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  # 3. Use free scales and 2 columns for clear separation
  facet_wrap(~ topic, scales = "free", ncol = 2) +
  coord_flip() +
  scale_x_reordered() +
  labs(title = NULL, x = NULL, y = NULL) +
  theme_minimal() +
  theme(
    panel.spacing.x = unit(4, "lines"),
    panel.spacing.y = unit(4, "lines"),
    strip.text = element_text(size = 11, face = "bold")
  )

# 4. Wider output for interactive clarity
print(ggplotly(p_topic_sent, width = 1200, height = 900))

# 5. Calculate sentiment frequency per topic
# We use all_topic_words from Step 3 to ensure full coverage
heatmap_data <- all_topic_words %>%
  inner_join(nrc, by = c("term" = "word"), relationship = "many-to-many") %>%
  group_by(topic, sentiment) %>%
  summarise(score = sum(beta), .groups = "drop")

# 6. Create the Heatmap
p_heatmap <- ggplot(heatmap_data, aes(x = factor(topic), y = sentiment, fill = score)) +
  geom_tile(color = "white") +
  # Use a red/orange scale similar to your reference Figure 6
  scale_fill_gradient(low = "white", high = "darkred") +
  theme_minimal() +
  labs(title = NULL, x = NULL, y = NULL, fill = NULL) +
  theme(axis.text.x = element_text(face = "bold"),
        axis.text.y = element_text(face = "bold"))

# 7. Render the plot
print(ggplotly(p_heatmap))

# ------------------------------------------------------------
# Step 4: Bigram Analysis
# ------------------------------------------------------------
speeches_bigrams <- speeches %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2)

bigrams_separated <- speeches_bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ")

bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !str_detect(word1, "\\d"),
         !str_detect(word2, "\\d"))

bigram_counts <- bigrams_filtered %>%
  count(word1, word2, sort = TRUE)

bigrams_united <- bigrams_filtered %>%
  unite(bigram, word1, word2, sep = " ")

p_bigram <- bigrams_united %>%
  count(bigram, sort = TRUE) %>%
  slice_max(n, n = 20) %>%
  mutate(bigram = reorder(bigram, n)) %>%
  ggplot(aes(n, bigram, fill = n)) +
  geom_col(show.legend = FALSE) +
  labs(title = NULL, x = NULL, y = NULL) +
  theme_minimal()

print(ggplotly(p_bigram))

# ------------------------------------------------------------
# Step 5: Network Graph of Co-occurring Words (REFINED)
# ------------------------------------------------------------
# 1. Filter for bigrams appearing 5 times or more
bigram_graph_data <- bigram_counts %>%
  filter(!word1 %in% c("graphene", "gan"),
         !word2 %in% c("graphene", "gan")) %>%
  filter(n >= 7) %>% # Changed to show bigrams appearing 5+ times
  graph_from_data_frame()

if (vcount(bigram_graph_data) > 0) {
  set.seed(2023)
  p_net_final <- ggraph(bigram_graph_data, layout = "fr") +
    # 2. Add edges with width and transparency based on frequency
    geom_edge_link(aes(edge_alpha = n, edge_width = n),
                   show.legend = TRUE,
                   arrow = grid::arrow(type = "closed", length = unit(.1, "inches")),
                   end_cap = circle(.07, "inches"),
                   color = "grey70") +
    # 3. Size nodes and add labels with overlap protection
    geom_node_point(color = "steelblue", size = 5) +
    geom_node_text(aes(label = name),
                   repel = TRUE,
                   size = 3.5,
                   fontface = "bold",
                   max.overlaps = 30) + # Limits labels to prevent clutter
    theme_void() +
    # 4. Clean title without the subtitle line
    labs(title = NULL, edge_alpha = NULL, edge_width = NULL)

  print(p_net_final)
} else {
  message("No bigrams found with n >= 5. Try a lower threshold.")
}
