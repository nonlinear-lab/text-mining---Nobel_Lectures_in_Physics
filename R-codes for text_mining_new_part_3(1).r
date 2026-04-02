# ============================================================
# TEXT MINING + TOPIC-SENTIMENT CORRELATION:
# Word Cloud aggregation, Topic-Sentiment NA drop,
# Emotional Arc per-president indexing
# ============================================================
# --- Load libraries ---
library(tidyverse)
library(tidytext)
library(tm)
library(ggplot2)
library(textdata)
library(topicmodels)
library(ggraph)
library(igraph)
library(plotly)
library(ggdendro)
# ------------------------------------------------------------
# Step 1: Load and Clean
# ------------------------------------------------------------
setwd("C:/R/physics1")
files <- list.files(pattern = "*.txt")
speeches <- map_df(files, function(f) {
  text_data <- readLines(f, encoding = "UTF-8", warn = FALSE)
  tibble(laureate = tools::file_path_sans_ext(f),
         text = paste(text_data, collapse = " "))
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
# Step 2: Speech Complexity Graph
# ------------------------------------------------------------
sentence_complexity <- speeches %>%
  mutate(sentences = str_count(text, "[.?!]"),
         word_count = str_count(text, "\\w+"),
         avg_sentence_length = word_count / sentences)

lexical_diversity <- tidy_speeches %>%
  group_by(laureate) %>%
  summarise(unique_words = n_distinct(word),
            total_words = n(),
            lexical_diversity = unique_words / total_words) %>%
  inner_join(sentence_complexity, by = "laureate")

p_complexity <- ggplot(lexical_diversity,
                       aes(x = avg_sentence_length, y = lexical_diversity, label = laureate)) +
  geom_point(aes(size = word_count), color = "steelblue", alpha = 0.7) +
  geom_text(vjust = -1, size = 3) +
  theme_minimal() +
  labs(title = NULL, x = "Ease of comprehension (Average words per sentence)", y = "Lexitical diversity(Type-token ratio)")

print(ggplotly(p_complexity))

# ------------------------------------------------------------
# Step 3: Emotional Arc
# ------------------------------------------------------------
narrative_arc <- tidy_speeches %>%
  inner_join(get_sentiments("bing"), by = "word") %>%
  group_by(laureate) %>%
  mutate(index = row_number() %/% 10) %>%
  count(laureate, index, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
  mutate(sentiment_score = positive - negative,
         pct_progress = (index / max(index)) * 100) %>%
  ungroup()

p_arc <- ggplot(narrative_arc, aes(x = pct_progress, y = sentiment_score, color = laureate)) +
  geom_smooth(method = "loess", se = FALSE, span = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  theme_minimal() +
  labs(title = NULL, x = NULL, y = NULL)

print(ggplotly(p_arc))

# ------------------------------------------------------------
# Step 4: Dendrogram
# ------------------------------------------------------------
dtm       <- tidy_speeches %>% count(laureate, word) %>% cast_dtm(laureate, word, n)
dtm_tfidf <- weightTfIdf(dtm)
m         <- as.matrix(removeSparseTerms(dtm_tfidf, 0.999))

dist_matrix <- dist(m, method = "euclidean")
fit         <- hclust(dist_matrix, method = "ward.D2")

# Save to PNG — bypasses the plotly HTML viewer device conflict
dendro_path <- "C:/R/speech/dendrogram.png"
png(filename = dendro_path, width = 1200, height = 700, res = 120)
par(mar = c(10, 4, 4, 2))
plot(as.dendrogram(fit),
     main = "", ylab = "", xlab = "", sub ="", cex  = 0.9)
dev.off()

# Auto-open the saved PNG in Windows
shell.exec(dendro_path)
message("Dendrogram saved and opened: ", dendro_path)


# ------------------------------------------------------------
# Step 5: Phrase Net
# ------------------------------------------------------------
connectors <- c("of", "the", "to", "in", "for", "with", "and", "our", "will")

phrase_net_data <- speeches %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter((word1 %in% connectors & !word2 %in% stop_words$word) |
           (word2 %in% connectors & !word1 %in% stop_words$word)) %>%
  count(word1, word2, sort = TRUE) %>%
  filter(n >= 10) # Increased from 2 to 3 to reduce clutter

if (nrow(phrase_net_data) > 0) {
  set.seed(42)
  p_net <- ggraph(graph_from_data_frame(phrase_net_data), layout = "nicely") +
    geom_edge_link(aes(edge_alpha = n, edge_width = n), edge_colour = "skyblue") +
    geom_node_point(size = 3, color = "darkblue") +
    # Increased max.overlaps to fix the ggrepel warning
    geom_node_text(aes(label = name), repel = TRUE, size = 3, max.overlaps = 50) +
    theme_void() + labs(title = NULL)

  print(p_net)
}
# -----------------------------------------------------------
# Step 6: Volatility (AFINN)
# ------------------------------------------------------------
volatility <- tidy_speeches %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(laureate) %>%
  summarise(mean_sent = mean(value), vol = sd(value))

p_vol <- ggplot(volatility, aes(x = mean_sent, y = vol, label = laureate)) +
  geom_point(color = "red", size = 3) +
  geom_text(vjust = -1) + theme_minimal() +
  labs(title = NULL)
print(ggplotly(p_vol))
