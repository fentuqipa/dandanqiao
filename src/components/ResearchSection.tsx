import {
  forwardRef,
  isValidElement,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  conferencePublications,
  journalPublications,
  underReviewPublications,
  workingPaperPublications,
  type TimelinePublication,
} from "../data/siteContent";

type PublicationFilter =
  | "all"
  | "journals"
  | "conferences"
  | "working-papers";

const publicationFilters: Array<{
  label: string;
  value: PublicationFilter;
}> = [
  { label: "All", value: "all" },
  { label: "Journals", value: "journals" },
  { label: "Conferences", value: "conferences" },
  { label: "Working Papers", value: "working-papers" },
];

const publicationHeadings: Record<PublicationFilter, string> = {
  all: "All Publications",
  journals: "Journal Publications",
  conferences: "Conference Publications",
  "working-papers": "Working Papers",
};

const INITIAL_RESULT_COUNT = 10;

function getSearchableText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }

  if (Array.isArray(node)) {
    return node.map(getSearchableText).join(" ");
  }

  if (isValidElement<{ children?: ReactNode }>(node)) {
    return getSearchableText(node.props.children);
  }

  return "";
}

export const ResearchSection = forwardRef<HTMLElement>(
  function ResearchSection(_, ref) {
    const [activeFilter, setActiveFilter] =
      useState<PublicationFilter>("all");
    const [searchQuery, setSearchQuery] = useState("");
    const [showAllPublications, setShowAllPublications] = useState(false);
    const normalizedQuery = searchQuery.trim().toLocaleLowerCase();

    const filteredPublications = useMemo(() => {
      let publications: TimelinePublication[];

      if (activeFilter === "journals") {
        publications = journalPublications;
      } else if (activeFilter === "conferences") {
        publications = conferencePublications;
      } else if (activeFilter === "working-papers") {
        publications = workingPaperPublications;
      } else {
        publications = [
          ...journalPublications.map((publication) => ({
            ...publication,
            category: "Journal" as const,
          })),
          ...conferencePublications.map((publication) => ({
            ...publication,
            category: "Conference" as const,
          })),
          ...underReviewPublications.map((publication) => ({
            ...publication,
            category: "Under Review" as const,
          })),
          ...workingPaperPublications.map((publication) => ({
            ...publication,
            category: "Working Paper" as const,
          })),
        ].sort((a, b) => Number(b.year) - Number(a.year));
      }

      return publications.filter((publication) =>
        getSearchableText(publication.text)
          .toLocaleLowerCase()
          .includes(normalizedQuery),
      );
    }, [activeFilter, normalizedQuery]);

    const visiblePublications = showAllPublications
      ? filteredPublications
      : filteredPublications.slice(0, INITIAL_RESULT_COUNT);
    const hasMorePublications =
      filteredPublications.length > INITIAL_RESULT_COUNT;
    const publicationHeading = publicationHeadings[activeFilter];

    return (
      <section id="research" ref={ref} className="research-section">
        <div className="section-kicker">
          <span>02</span>
          <p>Research</p>
          <div />
        </div>

        <div className="publication-controls">
          <label className="publication-search">
            <span className="publication-search-icon" aria-hidden="true" />
            <span className="sr-only">
              Search publications by title, author, or venue
            </span>
            <input
              type="search"
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              placeholder="Search title, author, or venue..."
            />
          </label>

          <div
            className="publication-tabs"
            role="tablist"
            aria-label="Publication type"
          >
            {publicationFilters.map((filter) => (
              <button
                key={filter.value}
                type="button"
                role="tab"
                aria-selected={activeFilter === filter.value}
                className={activeFilter === filter.value ? "active" : ""}
                onClick={() => setActiveFilter(filter.value)}
              >
                {filter.label}
              </button>
            ))}
          </div>
        </div>

        <div className="publication-results" role="tabpanel">
          {filteredPublications.length > 0 && (
            <PublicationTimeline
              title={publicationHeading}
              items={visiblePublications}
            />
          )}

          {filteredPublications.length === 0 && (
            <p className="publication-empty" role="status">
              {normalizedQuery
                ? `No publications match “${searchQuery.trim()}”.`
                : "No publications are available in this category yet."}
            </p>
          )}

          {hasMorePublications && (
            <div className="publication-more-wrap">
              <button
                type="button"
                className="publication-more-button"
                aria-expanded={showAllPublications}
                onClick={() =>
                  setShowAllPublications((isExpanded) => !isExpanded)
                }
              >
                {showAllPublications
                  ? "Hide Extra Publications"
                  : "Show More Publications"}
              </button>
            </div>
          )}
        </div>
      </section>
    );
  },
);

type PublicationTimelineProps = {
  title: string;
  items: TimelinePublication[];
};

function PublicationTimeline({ title, items }: PublicationTimelineProps) {
  return (
    <div className="publication-block">
      <h3 className="publication-heading">{title}</h3>

      <div className="publication-timeline">
        {items.map((item, index) => {
          const cardClassName = `publication-card${
            item.href ? " publication-card-link" : ""
          }${item.category ? " has-publication-type" : ""}`;

          const cardContent = (
            <>
              {item.category && (
                <span
                  className={`publication-type-tag publication-type-${item.category
                    .toLowerCase()
                    .replaceAll(" ", "-")}`}
                >
                  {item.category}
                </span>
              )}
              <p>{item.text}</p>
              <span>{item.year}</span>
            </>
          );

          return (
            <div className="publication-row" key={`${item.year}-${index}`}>
              <div className="timeline-marker">
                <span>{item.year}</span>
                <i />
              </div>

              {item.href ? (
                <a
                  className={cardClassName}
                  href={item.href}
                  target="_blank"
                  rel="noreferrer"
                >
                  {cardContent}
                </a>
              ) : (
                <article className={cardClassName}>{cardContent}</article>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
