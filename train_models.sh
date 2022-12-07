#!/bin/sh



for tl in 'premier_league' 'bundesliga' 'la_liga' 'ligue_1' 'serie_a'
do
    for ts in '2017-2018' '2018-2019' '2019-2020' '2020-2021' '2021-2022'
    do
        for tel in 'premier_league' 'bundesliga' 'la_liga' 'ligue_1' 'serie_a'
        do 
            for tes in '2017-2018' '2018-2019' '2019-2020' '2020-2021' '2021-2022'
            do 
                python3 model.py \
                --train_league $tl \
                --train_season $ts \
                --test_league $tel \
                --test_season $tes 
            done 
        done 
    done 
done