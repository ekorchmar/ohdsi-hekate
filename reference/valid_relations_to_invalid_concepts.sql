with
    rxne_all_relations as (
        select
            c1.vocabulary_id as vocabulary_id_1,
            c2.vocabulary_id as vocabulary_id_2,
            r.relationship_id
        from concept_relationship r
        join
            (
                select relationship_id
                from relationship
                where relationship_id < relationship.reverse_relationship_id
            ) n using (relationship_id)
        join
            concept c1
            on r.concept_id_1 = c1.concept_id
            and c1.vocabulary_id in ('RxNorm', 'RxNorm Extension')
        join
            concept c2
            on r.concept_id_2 = c2.concept_id
            and c2.vocabulary_id in ('RxNorm', 'RxNorm Extension')
        where
            r.invalid_reason is null
            and r.relationship_id not in ('Maps to', 'Concept replaced by')
            and (c1.invalid_reason is null) != (c2.invalid_reason is null)
    )
select count(*), vocabulary_id_1, vocabulary_id_2, relationship_id
from rxne_all_relations
group by vocabulary_id_1, vocabulary_id_2, relationship_id
