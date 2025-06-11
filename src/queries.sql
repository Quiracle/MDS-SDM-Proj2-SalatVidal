-- Get all papers authored by an author and their conference details

PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX pub: <http://example.org/publication/>
SELECT DISTINCT ?paper ?author ?conference ?venue WHERE{
    ?paper pub:hasAuthor ?author .
    
    ?author pub:hasName "E. Stefanakis" .
    
    ?paper pub:isPublishedIn ?proceeding .
    
    ?edition pub:hasProceedings ?proceeding;
             pub:hasVenue ?venue .
    
    ?conference pub:hasEdition ?edition .
}


-- Get the top 10 authors with the most citations
PREFIX pub: <http://example.org/publication/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?authorName (COUNT(DISTINCT ?citingPaper) AS ?numCitations)
WHERE {
  ?citingPaper a pub:Paper ;
               pub:cites ?citedPaper .
  ?citedPaper pub:hasAuthor ?author .
  ?author pub:hasName ?authorName .
}
GROUP BY ?authorName
ORDER BY DESC(?numCitations)
LIMIT 10