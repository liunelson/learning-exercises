# nature150 Data

This dataset was downloaded from the journal *Nature*'s 150th anniversary 
[immersive visualization](https://www.nature.com/immersive/d41586-019-03165-4/index.html) 
([pdf](https://www.nature.com/magazine-assets/d42859-019-00122-z/d42859-019-00122-z.pdf))
of a co-citation network of its publications.

| Filename | Headings | Size |
| --- | --- | --- |
| `cociteNodes.csv` | x, y, size, PubYear, HierCat, NatureID, Title | 88,283 rows | 
| `cociteEdges.csv` | source, target, path | 239,622 rows |

From the [accompanying article](https://www.nature.com/articles/d41586-019-03308-7):
> We extracted references for papers contained in the [Web of Science] publication database from 1900 to 2017, 
capturing close to 700 million citation relationships. 
We pinned subsequent analysis to the approximately 19 million articles that had at least one reference 
and one citation and that were published before 2010 (to give time for citations to accumulate). 
The resulting corpus integrated the discipline information for 38 million articles.
