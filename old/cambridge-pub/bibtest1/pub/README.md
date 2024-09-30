# MLG publications database

### Brief guide to the mlg.bib bibliography file for the mlg group.

The bibliography file mlg.bib contains the publications for the mlg
group. One of the purposes is that the web page for mlg publications
http://mlg.eng.cam.ac.uk/pub is automatically generated (currently
whenever the script to do this is run by Carl) from this bib file.
YOU ARE RESPONSIBLE FOR UPDATING THE FILE WITH YOUR PUBLICATIONS.

PLEASE make sure you don't check in modifications which are
syntactically incorrect - most editors will check the syntax for you.

Sharing of the file will be easiest if we all adhere to a few simple
conventions, which are described below:

1) cite key: use the following standard format: the three first
letters of each authors last name, the first letter capitalized,
followed by a two digit year. Example: "SaaTurRas10". If there are
more than four authors, just list three and add "etal". If there are
more than one ref which would get the same key, then use that key for
the first, and append "b", "c", etc for subsequent ones. If the last
name has less than 3 letters, then just use 2. Be consistent with
what's already in the file.

2) cat: assign each paper to zero or more subject categories, see
http://mlg.eng.cam.ac.uk/pub Use your judgement, eg don't categorize a
paper as mcmc just becasue it uses mcmc, only if it is about mcmc. The
labels are: gp clust gm mcmc ssl np approx bioinf ir rl time network
active neuro sigproc mvision mhearing fairness interpretability review
To assign to multiple categories use space seperated list, such as
cat = {gp gm review}.

3) title: protect important capitalization, ie as in title =
{Dirichlet's old friend {G}auss}.

4) author: the entries in the author field are used to automatically
generate the http://mlg.eng.cam.ac.uk/pub/authors author specific
reference lists, so you need to make sure 1) to write your name so that
the item gets found and included in your list, and 2) to make sure 
no-one else, with a similar name, ends up in your list; to help with 
this last issue, you can camouflage a name say "Nicholas R<!>oy", so
that it doesn't end up on Dan Roy's list ("Roy" is the key term).

5) abstract: if you use cut and paste, be carefull that ff, fl, fi and
quotation marks are copied correctly. Remove extraneous dash-es. Some
latex such as {\em example} is converted correctly, but not all (eg,
use <br> instead of \\). Use html formatting.

6) note: information from "note" field is typeset at the end of the
reference.

7) annote: the contents of this field is typeset as "comment: ..." in
a paragraph after the abstract. Use this for comments, html pointers to
suplementary material, code etc

8) url: give a url, usually for the pdf of the paper (this will be the
link followed from the title of the paper on the publications
page). For papers which are copyright, you should link to the official
(journal) page for the paper. For papers which we can distribute, put
a copy of the paper pdf in the directory /var/www/mlg/pub/pdf on
cbl-fs. The name of the file must be "cite key".pdf. In this case, you
should specify "url = {.}", the system will automatically link to the
right file. If in rare cases there is nothing relevant to link to,
then don't include a url.

9) doi: give the doi (Digital Object Identifier). The system will
create a link to http://dx.doi.org/DOI, where DOI is the doi.

All the papers will have anchors which are their cite keys. This is
useful, you can refer to a paper as
http://mlg.eng.cam.ac.uk/pub/#SaaTurRas10 You can also link to other
papers from the abstract or annote fields, in this case use something
like <a href="/pub/#SaaTurRas10">paper</a>.

  Carl  (2015-09-24)

### Updating the git repo

For those unfamiliar with git, here's a quick way to update the mlg.bib file.
There are two main ways:

1. Through the web interface.  
  You can edit the mlg.bib file directly through GitHub's web interface.
  When done you'll need to add a commit title at the bottom,
  such as "add <citekey>" and click on "commit changes". The changes should
  be immediately applied.

2. By cloning the repo locally.
  * Go to a suitable directory and type
    `git clone https://github.com/cambridge-mlg/publications`
    (you'll need your GitHub username and password).
    
  * This will create a directory called "publications" with the mlg.bib file.
  
  * Edit the mlg.bib file normally.
  
  * Go to the publications folder and type the following:
    ```
    git add -u
    git commit -m "Add my publication"
    git push
    ```
    You will need the GitHub username and password again.
    If the push was successfull the changes should be immediately
    visible on GitHub.
    
  * You can now safely delete the publications folder on your machine.
