customer_context = """
A customer (individual advertiser) uses the self-service portal to book an ad in a publication. The chatbot can guide a customer through these steps to book an advertisement:

Login/Signup: The customer logs into the ad portal with their account. If new, they sign up and then log in to access the dashboard.

Start New Ad Booking: From the customer dashboard, they choose the option to “Book a New Ad”. This initiates the ad booking workflow.

Select Ad Type & Category: The customer first selects the type of ad (e.g. Classified Text or Display Ad) and the appropriate category for the ad (for example, Matrimonial, Recruitment, Real Estate, etc.)
releasemyad.com
. Based on the category chosen, the system shows available publications and any relevant offers or packages
releasemyad.com
.

Choose Publication/Edition: Next, the customer chooses the publication (newspaper/magazine) and the edition or city in which the ad should appear
releasemyad.com
. If the portal supports multiple publications, the user selects one (or a package covering multiple editions) that fits their needs and budget.

Compose Ad Content: The customer then composes the advertisement content. For a text ad, they type in the ad copy (and can apply text enhancements like bold or background color if available); for a display ad, they can upload the artwork or use a template
releasemyad.com
. The portal may provide a character count or a preview at this stage so the user can see how the ad will look.

Specify Ad Details: The user provides any additional details required. This can include specifying the ad size or dimensions (for display ads), choosing whether the ad is in color or black & white, and any position preferences (like a specific page or section, if applicable)
dkatia.com
. They also enter the contact information or advertiser name if not already on file (this ensures the correct name appears on the booking and invoice).

Select Publication Date(s): The customer selects the date or dates on which the ad should run. They can pick a single date or multiple insertion dates, depending on their need and the publication’s schedule
releasemyad.com
. The system will check availability for the chosen date(s) and prevent double-booking of the same space if that’s a concern for display ads.

Review Price and Preview: The portal now shows a summary of the booking with a preview of the ad and the calculated cost. Pricing is typically based on factors like ad type (text vs display), size/word count, number of insertions, and any package discounts
releasemyad.com
releasemyad.com
. The customer should review all details here to ensure accuracy.

Make Payment: After confirming the details, the customer proceeds to payment. They choose from available payment options (credit/debit card, net banking, digital wallet, etc.) and pay the ad booking amount
releasemyad.com
. The platform may support online payment as well as offline modes in some cases (e.g. cash/cheque at an office, though online is most common).

Confirmation: Once payment is successful, the booking is confirmed. The customer receives a confirmation message or booking reference number on-screen (and likely via email/SMS). The ad is now scheduled to be published on the chosen date, subject to any internal approval process. The user can view the booking status in their account (it might show as “Pending Approval” or “Confirmed”).

(The system now internally handles the order – for example, it may await manager approval or go straight to scheduling depending on the ad type. But from the customer’s perspective, the booking flow is complete and their ad will be published on the scheduled date
releasemyad.com
.)
"""

agency_context = """
Advertising agencies use the portal to book ads on behalf of their clients. The flow is similar to the customer flow, but with a few additional features to accommodate agency needs:

Agency Login: The agency user logs into the portal using their agency credentials. Agencies typically have a special account that allows managing multiple client bookings. The portal supports a dedicated Agency Login for direct online ad booking
dkatia.com
.

Dashboard Navigation: On the agency dashboard, the agent selects the option to create a new ad booking. They might first choose or input the client/advertiser name for whom the ad is being booked (if the system requires linking the booking to a specific client record).

Select Ad Type & Category: The agency selects the advertisement type (classified text, classified display, or display ad) and the relevant category (e.g. jobs, notices, etc.), just like a regular customer would
releasemyad.com
. The interface may allow the agency to also choose the media (publication and edition) at this stage, including multi-publication packages if needed (the portal supports multiple publications and editions in one place
dkatia.com
).

Enter Ad Details: The agent fills in all ad details. This includes writing the ad copy or uploading creative content, selecting ad size/format if applicable, and specifying any position or page requests for display ads
dkatia.com
. Essentially, the agency provides the same information as an individual customer would, but they can do it on behalf of a client. The portal may also allow bulk ad booking – for instance, booking the same ad in multiple publications or dates in one go – which is a feature useful for agencies
dkatia.com
.

Schedule Dates and Editions: The agency selects the publication and the run date(s) for the advertisement. They can book the ad in one or multiple editions and on one or multiple dates as required. The system will show available slots and prevent conflicts (e.g., it checks space availability so no duplicate bookings occur for the same slot)
dkatia.com
.

Review and Confirm: The agent reviews the booking summary, which shows the ad preview and pricing. Agencies often get negotiated rates or can apply any client-specific discount codes at this stage. The portal might generate an instant quote/invoice for the ad based on standard rates
dkatia.com
. The agent ensures all details are correct.

Payment or Invoice: The agency then completes the booking. Depending on the setup, the agency might pay via the portal (just like a customer) or opt to be invoiced. Many agency bookings use a credit system or monthly billing rather than immediate online payment. The portal supports instant invoice generation for agency bookings (especially for classified ads)
dkatia.com
. So an agency could either download the invoice to pay offline or settle via their credit account.

Booking Confirmation: After submission, the booking is recorded in the system. The agency receives a confirmation reference for the ad booking. If payment was made online, it’s confirmed immediately; if not, the booking might be marked as “Pending Payment” or “Payment via Invoice”. In both cases, the ad is reserved for the chosen date/edition. The agency can see this booking in their dashboard along with all their client bookings.

(Agency users benefit from additional functionality in the system, such as bulk bookings and integrated invoicing
dkatia.com
. Their bookings might still go through an internal approval by the publisher’s management, especially if special discounts were applied, but the agency’s flow for inputting the booking is as outlined above.)
"""
