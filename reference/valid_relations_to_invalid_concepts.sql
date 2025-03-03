with n as (
    select relationship.relationship_id
    from relationship
    where relationship.relationship_id < relationship.reverse_relationship_id
),

rxne_all_relations as (
    select
        c1.vocabulary_id as vocabulary_id_1,
        c2.vocabulary_id as vocabulary_id_2,
        r.relationship_id
    from concept_relationship as r
    inner join
        n on r.relationship_id = n.relationship_id
    inner join
        concept as c1
        on
            r.concept_id_1 = c1.concept_id
            and c1.vocabulary_id in ('RxNorm', 'RxNorm Extension')
    inner join
        concept as c2
        on
            r.concept_id_2 = c2.concept_id
            and c2.vocabulary_id in ('RxNorm', 'RxNorm Extension')
    where
        r.invalid_reason is null
        and r.relationship_id not in ('Maps to', 'Concept replaced by')
        and (c1.invalid_reason is null) != (c2.invalid_reason is null)
)

select
    a.vocabulary_id_1,
    a.vocabulary_id_2,
    a.relationship_id,
    count(*) as cnt
from rxne_all_relations as a
group by a.vocabulary_id_1, a.vocabulary_id_2, a.relationship_id
