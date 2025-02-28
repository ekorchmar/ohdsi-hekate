-- List of distinct relationships between RxNorm and RxNorm extension classes
with
class_relationship as (
    select
        r.relationship_id, c1.concept_class_id as concept_class_id_1, c2.concept_class_id as cid2
    from concept c1
    join
        concept_relationship r
        on c1.concept_id = r.concept_id_1
        and c1.vocabulary_id in ('RxNorm', 'RxNorm Extension')
        and r.invalid_reason is null
        and c1.invalid_reason is null
    join
        concept c2
        on c2.invalid_reason is null
        and c2.vocabulary_id in ('RxNorm', 'RxNorm Extension')
        and c2.concept_id = r.concept_id_2
)
select concept_class_id_1, relationship_id, cid2, count(*)
from class_relationship
group by relationship_id, concept_class_id_1, cid2
;
