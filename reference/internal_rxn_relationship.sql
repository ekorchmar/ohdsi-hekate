-- List of distinct relationships between RxNorm and RxNorm extension classes
with
class_relationship as (
    select
        r.relationship_id,
        c1.concept_class_id as concept_class_id_1,
        c2.concept_class_id as cid2
    from concept as c1
    inner join
        concept_relationship as r
        on
            c1.concept_id = r.concept_id_1
            and c1.vocabulary_id in ('RxNorm', 'RxNorm Extension')
            and r.invalid_reason is null
            and c1.invalid_reason is null
    inner join
        concept as c2
        on
            c2.invalid_reason is null
            and c2.vocabulary_id in ('RxNorm', 'RxNorm Extension')
            and r.concept_id_2 = c2.concept_id
)

select
    cr.concept_class_id_1,
    cr.relationship_id,
    cr.cid2,
    count(*) as cnt
from class_relationship as cr
group by cr.relationship_id, cr.concept_class_id_1, cr.cid2;
