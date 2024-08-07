# Netflix Dataset Analysis (Using SQL)


# 1. Number of Movies and TV Shows
SELECT type,
COUNT(*) AS count
FROM NetflixTable
WHERE type IN ('MOVIE', 'SHOW')
GROUP BY type;


# 2. Release on Movies and TV Shows in different Decades
SELECT
    CASE
        WHEN release_year BETWEEN 1900 AND 1999 THEN '1900s-1999'
        WHEN release_year BETWEEN 2000 AND 2009 THEN '2000s-2009'
        WHEN release_year BETWEEN 2010 AND 2019 THEN '2010s-2019'
        WHEN release_year BETWEEN 2020 AND 2029 THEN '2020s-2021'
    END AS decade,
    type,
    COUNT(*) AS count
FROM NetflixTitles
GROUP BY decade, type;


# 3. Average IMDb score for movies
SELECT AVG(CAST(imdb_score AS DECIMAL(4, 2)) ) AS average_imdb_score
FROM NetflixTable
WHERE type = 'MOVIE';


# 4. Top 10 Movies with the Highest IMDb Score:
SELECT title, imdb_score
FROM NetflixTable
WHERE type = 'MOVIE'
ORDER BY imdb_score DESC
LIMIT 10;


# 5. Bottom 10 Movies with the Lowest IMDb Score:
SELECT title, imdb_score
FROM NetflixTable
WHERE type = 'MOVIE' AND imdb_score IS NOT NULL
ORDER BY imdb_score
LIMIT 10;


# 6. Top 10 TV Shows with the Most Seasons:
SELECT title, seasons
FROM NetflixTable
WHERE type = 'SHOW'
ORDER BY CAST(seasons AS SIGNED) DESC
LIMIT 10;


#7. Top 10 Movies with the Longest Runtime:
SELECT title, runtime
FROM NetflixTable
WHERE type = 'MOVIE' AND runtime IS NOT NULL
ORDER BY CAST(runtime AS SIGNED) DESC
LIMIT 10;


# 8. Top 10 most popular TV shows (by TMDB popularity score):
SELECT title, tmdb_popularity
FROM NetflixTable
WHERE type = 'SHOW'
ORDER BY tmdb_popularity DESC
LIMIT 10;


# 9. Top 10 Movie with the highest IMDb score in the 'drama' genre:
SELECT title, imdb_score
FROM NetflixTable
WHERE type = 'MOVIE'
AND genres LIKE '%drama%'
ORDER BY imdb_score DESC
LIMIT 10;


# 10. Top 10 most common production countries:
SELECT production_countries, COUNT(*) AS count
FROM NetflixTable
GROUP BY production_countries
ORDER BY count DESC
LIMIT 10; 


# 11. Count of TV shows by age certification:
SELECT age_certification, COUNT(*) as show_count
FROM NetflixTable
WHERE type = 'SHOW'
GROUP BY age_certification;

