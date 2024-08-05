#!/bin/bash
openout_any=a
export openout_any
echo 'Making the html files'

CATEGORIES="gp clust gm mcmc ssl np approx bioinf ir rl time review network active neuro sigproc mvision mhearing deep interpretability fairness nlp"

rm -fr ../hugo/layouts/partials/ref-*.html

for CAT in $CATEGORIES; do
    # Extract all entries of a category
    bibtool -- preserve.key.case=on -- 'select{cat "'"$CAT"'"}' ../mlg.bib -o $CAT.bib
    # Make the html version
    ./bib2html mlg2 -o ../hugo/layouts/partials/ref-$CAT.html $CAT.bib 
    rm $CAT.bib
done

YEARS="2022 2021 2020 2019 2018 2017 2016 2015 2014 2013 2012 2011 2010 2009 2008 2007 2006 2005 2004 2003 2002 2001 2000\\|199"

for YEAR in $YEARS; do
    # Extract all entries of a category
    bibtool -- preserve.key.case=on -- 'select{year "'"$YEAR"'"}' ../mlg.bib -o $YEAR.bib
    # Make the html version, NICEYEAR strips out backslash from year for past milenea items
    NICEYEAR=${YEAR/\\/}
    ./bib2html mlg -o ../hugo/layouts/partials/ref-$NICEYEAR.html $YEAR.bib 
    rm $YEAR.bib
done

AUTHORS="Allingham Antor Ashman Balog Bauer Bhatt Borgwardt Brati Bronskill Bruinsma Bui Burt Calliess Cheema Wenlin Yutian Clarke Cunningham Davies Daxberger Dutordoir Duvenaud Deisenroth Dziugaite Eaton Flamich Foong Fortuin Frellsen Frigola Gael Gal Garriga-Alonso Hong Ghahramani Gordon Shixiang Heaukulani Heller Hoffman Houlsby Hron Ferenc Ialongo Janz Kilbertus Knowles Kok Krasheninnikov Krueger Lacoste Lalchand Langosco Yingzhen Likhosherstov Lloyd Lobato Lomeli Lopez-Paz Markou Matthews McAllister McHutchon Mohamed Nalisnick Navarro Oldewage Orbanz Ortega Palla Peharz Pinsler Quadrianto Rasmussen Requeima Rowland Roy Yunus cibior Shah Siddiqui Steinruecken Swaroop Terenin Tebbutt Tobar Tripp Turner Rojas-Carulla Ryan Weller Sinead Snelson Wilson Wilk Zilka"

for AUTH in $AUTHORS; do
    # Extract all entries of a category
    bibtool -- preserve.key.case=on -- 'select{author "'"$AUTH"'"}' ../mlg.bib -o $AUTH.bib
    # Make the html version
    ./bib2html mlg2 -o ../hugo/layouts/partials/ref-$AUTH.html $AUTH.bib 
    rm $AUTH.bib
done
