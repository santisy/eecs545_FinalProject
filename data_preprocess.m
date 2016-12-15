p = load('movielensFull.mat');
m_data = p.movieLens;
length_count = 0;
movie_full = [];
for i = 1:5
    length_count = length_count + size(m_data{i},1);
    movie_full = [movie_full;m_data{i}];
end
max(unique(movie_full(:,1)))
max(unique(movie_full(:,2)))

train_vec = movie_full(1:900000,1:3);
probe_vec = movie_full(900001:end,1:3);