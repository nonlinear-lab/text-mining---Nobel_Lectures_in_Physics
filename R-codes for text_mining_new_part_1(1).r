# --- Load libraries (Your original list) ---
library(tidyverse)
library(tidytext)
library(tm)
library(wordcloud) # We are using the original one to avoid the 'markdown' error
library(topicmodels)
library(plotly)

# ------------------------------------------------------------
# Step 1: Load and Clean
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
# Step 2: Word Frequency per laureate
# ------------------------------------------------------------
word_freq <- tidy_speeches %>% count(president, word, sort = TRUE)

p_freq <- word_freq %>%
  group_by(president) %>%
  slice_max(n, n = 5) %>%
  ungroup() %>%
  ggplot(aes(x = reorder_within(word, n, president), y = n, fill = president)) +
  geom_col(show.legend = TRUE) + # Legend moved to bottom to save width
  facet_wrap(~ president, scales = "free", ncol = 2) +
  coord_flip() +
  scale_x_reordered() +
  labs(title = NULL, x = NULL) +
  theme_minimal() +
  theme(
    # 1. Move legend to bottom to maximize horizontal plot space
    legend.position = "bottom",
    # 2. Increase vertical spacing to prevent row collision
    panel.spacing.y = unit(1.5, "lines"),
    # 3. Increase horizontal spacing between columns
    panel.spacing.x = unit(2, "lines"),
    # 4. Rotate x-axis text to prevent horizontal overlap of numbers
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 8, face = "bold")
  )

# Use explicit width and height to prevent 'squeezing' in the viewer
print(ggplotly(p_freq, width = 1000, height = 1000))

# ------------------------------------------------------------
# Step 3: Word Cloud
# ------------------------------------------------------------
word_freq_total <- word_freq %>%
  group_by(word) %>%
  summarise(total = sum(n)) %>%
  arrange(desc(total))

# Open a new window to prevent it from being hidden
if(.Platform$OS.type == "windows") { windows() } else { x11() }

set.seed(123)
# Note: scale is reduced to c(3, 0.3) so words actually fit!
wordcloud(
  words        = word_freq_total$word,
  freq         = word_freq_total$total,
  max.words    = 80,
  random.order = FALSE,
  colors       = brewer.pal(8, "Dark2"),
  scale        = c(3, 0.3)
)
title("")

# ------------------------------------------------------------
# Step 4: Topic Modeling (LDA)
# ------------------------------------------------------------
# ------------------------------------------------------------
# Step 4: Topic Modeling (LDA) - ADJUSTED
# ------------------------------------------------------------
# 1. Create the DTM
dtm <- tidy_speeches %>%
  count(president, word) %>%
  cast_dtm(president, word, n)

# 2. Set k=4 to match your 4-panel figure
# IMPORTANT: set.seed(2023) ensures the topics don't change every time you run it
set.seed(1200)
lda <- LDA(dtm, k = 4, method = "VEM")

# 3. Process the results for visualization
p_topic <- tidy(lda, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 12) %>% # Increased to 12 to match the density of your figure
  ungroup() %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free", ncol = 2) +
  coord_flip() +
  scale_x_reordered() +
  labs(title = NULL, x = NULL, y = "Beta (Topic Contribution)") +
  theme_minimal() +
  theme(
    panel.spacing.x = unit(2, "lines"),
    panel.spacing.y = unit(2, "lines"),
    axis.text.y = element_text(size = 9),
    strip.text = element_text(size = 10, face = "bold")
  )

# Render the interactive plot
print(ggplotly(p_topic, width = 1000, height = 800))


# ------------------------------------------------------------
# Step 5: Sentiment Analysis (Corrected Top 5 Per Laureate)
# ------------------------------------------------------------
nrc <- get_sentiments("nrc")

p_sent <- tidy_speeches %>%
  inner_join(nrc, by = "word", relationship = "many-to-many") %>%
  count(president, sentiment) %>%
  # 1. GROUP BY PRESIDENT FIRST so each laureate gets 5 bars
  group_by(president) %>%
  slice_max(n, n = 5, with_ties = FALSE) %>%
  ungroup() %>%
  ggplot(aes(reorder_within(sentiment, n, president), n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  # 2. Use 2 columns to prevent horizontal overlapping
  facet_wrap(~ president, scales = "free", ncol = 2) +
  coord_flip() +
  scale_x_reordered() +
  labs(title = NULL, x = NULL, y = NULL) +
  theme_minimal() +
  theme(
    # 3. Increase spacing to prevent titles and axes from colliding
    panel.spacing.x = unit(4, "lines"),
    panel.spacing.y = unit(2.5, "lines"),
    # 4. Tilt numbers to avoid vertical overlap
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    strip.text = element_text(size = 9, face = "bold")
  )

# Use explicit dimensions to ensure clarity in the interactive viewer
print(ggplotly(p_sent, width = 1100, height = 1100))
